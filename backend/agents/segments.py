import json
import httpx
# from pathlib import Path

class SegmentsAgent:
    def __init__(self, httpx_client, llm_client, volume_estimator_endpoint = 'http://192.168.1.191:5000/predict'):
        self.httpx_client = httpx_client
        self.llm_client = llm_client
        self.volume_estimator_api = volume_estimator_endpoint
        self.prompt = """CRITICAL: This image shows a Food Item. Focus on main food item and ignore rest. You must identify main ingredient in Food Item.
CONTEXT:
- Image of Food Item
- Volume of THIS ITEM: {volume_ml:.3f} ml

YOUR TASK:
1. Identify the main ingredient in food item that contibutes major calorie. Focus on uncertainties that SIGNIFICANTLY affect calorie count.
2. Based on the most significant uncertainty, formulate ONLY ONE critical question to ask the user. This question should aim to resolve the biggest potential calorie difference.

ASK YOURSELF:
1. Can I clearly identify what this food IS?
- If it's white liquid: Is it milk, yogurt, or cream?
- If it's protein: Chicken, paneer, tofu, or egg?

2. What is the single biggest point of confusion that impacts calories?
- Could this be paneer OR tofu? (This is a huge calorie difference, a great question to ask).
- Could this be a sugar-free dessert OR a full-sugar dessert? (Also a great question).

DO NOT worry about:
- Cooking oil type
- Spices or garnishes
- Type of grains (e.g., basmati vs. sona masoori rice)
- Minor brand differences

RESPONSE FORMAT (JSON):
{{
"food_name": "best guess for THIS ITEM only",
"confidence": 0.0-1.0,
"major_uncertainties": [
    "A list of reasons for the uncertainty, focusing on high-calorie differences."
],
"most_important_question": "The single most critical question to resolve calorie ambiguity. Should be an empty string if confidence is > 0.94.",
"ambiguity_flag": true/false
}}

CONFIDENCE LEVELS:
- 0.9-1.0: I know exactly what this is. No question needed.
- 0.8-0.9: I know the food but there's a minor variety uncertainty. A question might be useful but not critical.
- 0.7-0.8: Could be 2-3 different foods. A question is necessary.
- <0.7: Very unclear. A question is necessary.

Set ambiguity_flag TRUE if:
- You cannot clearly identify the food.
- This could be 2+ foods with a 50+ calorie difference per 100g.

EXAMPLES of how to generate the single question:

Uncertainty: "This could be paneer (high protein, high cal) or tofu (lower cal)"
Resulting JSON field: `"most_important_question": "Is this paneer or tofu?"`

Uncertainty: "I cannot tell if this is ketchup (high sugar) or tomato chutney (lower sugar)"
Resulting JSON field: `"most_important_question": "Is this ketchup or a homemade tomato chutney?"`

Uncertainty: "Cannot tell if this is a fried pakora (high cal) or a steamed idli (low cal)"
Resulting JSON field: `"most_important_question": "Is this item fried or steamed?"`

Respond ONLY with the JSON object."""
    async def call_volume_estimation_api(self, image, format='base_64', plate_diameter = 0.3):
        print(f"Calling volume estimation API for image.")
        try:
            if format == 'base_64':
                payload = {
                    'img': image,
                    'plate_diameter':plate_diameter or 0.3
                }
                response = await self.httpx_client.post(self.volume_estimator_api, json=payload)
            else:
                files = {'img': image}
                data = {'format': format, 'plate_diameter': plate_diameter}
                response = await self.httpx_client.post(self.volume_estimator_api, files=files, data=data)
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            print(f"An error occurred while calling the Volume Estimation Endpoint: {e}")
            return None

    async def analyze_food_image(self, image_base64, volume_ml: float) -> dict:
        """Analyze a single food segment image using Gemini VLM"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt.format(volume_ml=volume_ml)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
                
            # Validate we got a specific food, not a meal description
            # food_name = result.get('food_name', 'Unknown')
            # if any(word in food_name.lower() for word in ['meal', 'plate', 'dish with', 'and', 'rice meal', 'with dal']):
            #     print(f"  WARNING: VLM returned meal description instead of specific item: {food_name}")
            #     print(f"  Attempting to extract specific food...")
            #     # Try to force it to be more specific
            #     result['ambiguity_flag'] = True
            #     result['confidence'] = min(result.get('confidence', 0.5), 0.7)
            
            # Ensure required fields
            if 'food_name' not in result:
                result['food_name'] = 'Unknown food'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'ambiguity_flag' not in result:
                result['ambiguity_flag'] = True
            if 'major_uncertainties' not in result:
                result['major_uncertainties'] = []
            if 'most_important_question' not in result:
                result['most_important_question'] = ""
                
            return result
            
        except json.JSONDecodeError as e:
            print(f"Warning: VLM returned invalid JSON: {e}")
            return {
                "food_name": "Unknown food",
                "confidence": 0.0,
                "ambiguity_flag": True,
                "what_i_see": "Error parsing response",
                "what_i_cannot_determine": [],
                "assumptions_i_am_making": [],
                "error": str(e)
            }
        except Exception as e:
            print(f"Error during VLM API call: {e}")
            return {
                "food_name": "Unknown food",
                "confidence": 0.0,
                "ambiguity_flag": True,
                "what_i_cannot_determine": [],
                "assumptions_i_am_making": [],
                "error": str(e)
            }

    async def run_vlm_analysis(self, volumes):
        """Analyze all food segments using VLM"""
        if volumes is None or len(volumes) == 0:
            return None
        print("\n" + "="*60)
        print("PART 1: VLM FOOD IDENTIFICATION")
        print("="*60)
        
        print(f"\nFound {len(volumes['segments'])} food segments to analyze\n")
        
        for i, segment in enumerate(volumes['segments'], 1):
            volume_ml = segment['volume_ml']
            image_base64 = segment['image_base64']
            
            print(f"[{i}/{volumes['total_segments']}] Analyzing Segment {segment['segment_id']}...")
            print(f"  Volume: {volume_ml:.3f} ml")
            
            # Analyze with VLM
            analysis = await self.analyze_food_image(image_base64, volume_ml)
            segment.update(analysis)
            print(f"  Identified: {analysis.get('food_name', 'Unknown')}")
        print("="*60)
        print("VLM Analysis Complete")
        print("="*60 + "\n")
        return volumes
    
    async def __call__(self, image, format=None):
        if format:
            return await self.run_vlm_analysis(await self.call_volume_estimation_api(image, format))
        else:
            return await self.run_vlm_analysis(await self.call_volume_estimation_api(image))

def main():
    import time
    from openai import OpenAI
    import sys
    import os
    import httpx, base64
# pyrefly: ignore [missing-import]

    from dialogue_agent import DialogueAgent
    from dotenv import load_dotenv
    
    load_dotenv()
    BASE_DIR = os.path.dirname(__file__)
    input_image_path = BASE_DIR+"/uploads/test2.png"
    

    httpx_client = httpx.Client(timeout=150)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise("Error: API key not found in config.py")
        sys.exit(1) # Changed from 'return' to 'sys.exit(1)'

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    # Step 1: Call Volume Estimation API
    # Step 1: Call Volume Estimation API
    print("STEP 1: Volume Estimation")
    print("-" * 60)
    agent = SegmentsAgent(httpx_client, llm_client)
    volume_start_time = time.time()
    with open(input_image_path, 'rb') as image:
        # volumes = call_volume_estimation_api(httpx_client, image, format='file')
        image_b64 = base64.b64encode(image.read()).decode("utf-8")
        volumes = agent(image_b64, format='base_64')
    volume_end_time = time.time()
    print(f"Volume Estimation API call took: {volume_end_time - volume_start_time:.2f} seconds")
    print([[result['volume_ml'], result['food_name']] for result in volumes['segments']])
    # with open('agents/vlm_output.json', 'w', encoding='utf-8') as f:
    #     json.dump(volumes, f, indent=2, ensure_ascii=False)
    # with open('agents/vlm_output.json', 'r', encoding='utf-8') as f:
    #     volumes = json.load(f)
    
    # dialogue_start_time = time.time()
    # dialogue_agent = DialogueAgent(client)
    # dialogue_agent.confirm_analysis(volumes)
    # dialogue_end_time = time.time()
    # print(f"Dialogue Agent took: {dialogue_end_time - dialogue_start_time:.2f} seconds")
    
    # with open('agents/justified_answers.json', 'w', encoding='utf-8') as f:
    #     json.dump(volumes, f, indent=2, ensure_ascii=False)
    # # Step 4: Save Final Output
    # FINAL_OUTPUT_FILE = "final_confirmed_output.json"
    
    # print("\n" + "="*60)
    # print("PIPELINE COMPLETE!")
    # print("="*60)
    # print(f"Final output: {FINAL_OUTPUT_FILE}")
    # print("="*60 + "\n")

if __name__ == "__main__":
    main()
    # Note this was intended for testing purpose only before async shift for sadly fast api calls.