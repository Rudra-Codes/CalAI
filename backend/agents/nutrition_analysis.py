# from tavily import TavilyClient
import json

class NutritionAgent:
  def __init__(self, tavily_client, llm_client):
    # self.volumes = volumes
    self.tavily_client = tavily_client
    self.llm_client = llm_client

    self.density_parser_prompt = """
You are an information extraction assistant.

From the provided context, extract:
1. Density of the food item.
2. Calories (cal), protein (g), fat (g), and food weight (g).

Rules:
- Use only values explicitly stated in the context.
- If multiple values exist, select the single best match for the food item.
- Convert density to g/mL when an explicit density value is provided in any convertible unit.
- If weight is present, normalize calories, protein, and fat to per-gram values using:
  per_gram_value = total_value / weight_in_grams
- Extract and derive information in required units as mentioned.
- If a nutrient value is missing, return null for that nutrient.
- If no explicit density is provided, return null.

Output only valid JSON:

{
  "density": <number|null>,
  "calories_per_g": <number|null>,
  "protein_per_g": <number|null>,
  "fat_per_g": <number|null>
}
"""
    self.response_prompt = """You are an intelligent food and nutrition assistant. You can:
- Answer general food and nutrition questions
- Analyze food images
- Provide meal recommendations
- Compare foods and meals
- Give dietary advice
- Access past nutrition calculations if exists

CONTEXT AVAILABLE TO YOU:
{context_summary}

INSTRUCTIONS:
1. When user asks about "my meal", "the food I ate", "what I calculated" - refer to the calorie calculations above
2. You have access to DETAILED nutritional breakdowns - use them!
3. If comparing foods, use the exact values from calculations
4. Be specific with numbers when you have them
5. If you see an image, describe it and answer accordingly
6. Do not mention about context.
7. If you do not have required calculations, estimate and mention it.

IMPORTANT: The calculations may contain exact values for calories, protein, carbs, and fat. Use these precise numbers in your responses!

Respond naturally and conversationally."""
  async def density_calculator(self, volumes):
  # context = ""
    for volume in volumes['segments']:
      context = ""
      web_search = (await self.tavily_client.search(f"What is density of {volume['food_name']}", max_results=3))['results']
      # for contexts in web_search:
      #     context += contexts["content"] + "\n"
      context += '\n'.join([contexts['content'][:1000] for contexts in web_search]) + '\n'
      web_search = (await self.tavily_client.search(f"What are calories, proteins and fats in per gram of {volume['food_name']}", max_results=3))['results']
      
      context += '\n'.join([contexts['content'][:1000] for contexts in web_search])
      print("Nutrition search completed")
      # print(context)
      try:
          response = await self.llm_client.chat.completions.create(
              model="openai/gpt-oss-20b",
              # model="openai/gpt-oss-120b", # Neglected as giving same output upper one is cheap and fast.
              messages=[
                  {
                      "role": "system",
                      "content": self.density_parser_prompt
                  },
                  {"role": "user", "content": context+f"What is density and calories, proteins, fats in {volume['food_name']}?"}
              ],
              response_format={"type": "json_object"},
              temperature=0.2
          )

          result = json.loads(response.choices[0].message.content)
          print(result)
          # volume.update(result)
          density = result.get("density")
          if density not in (None, ""):
            volume['density'] = float(density)
            volume['weight_in_g'] = float(density)*float(volume['volume_ml'])
            if result.get('calories_per_g') is not None:
              volume['calories'] = volume['weight_in_g']*result['calories_per_g']
            if result.get('protein_per_g') is not None:
              volume['protein'] = volume['weight_in_g']*result['protein_per_g']
            if result.get('fat_per_g') is not None:
              volume['fat'] = volume['weight_in_g']*result['fat_per_g']
          # print(volume['density'], volume['weight_in_g'], volume['calories'], volume['proteins'], volume['fat'])
      except Exception as e:
        print(f"Parsing density failed: {e}")
  
  async def query_food_assistant(self, volumes, user_query, image_base64=None):
    """
    Parses the food segments dictionary, extracts non-null nutritional 
    information, and sends a formatted prompt to the OpenAI API.
    """
    keys_to_pass = ['segment_id', 'food_name', 'volume_ml', 'weight_in_g', 'calories', 'protein', 'fat']
    context_string = "\n\n".join(
      "\n".join(f'{k}- {segment[k]}' for k in keys_to_pass if k in segment) for segment in volumes['segments']
    )

    # Execute the API call
    try:
      response = await self.llm_client.chat.completions.create(
          model= "meta-llama/llama-4-scout-17b-16e-instruct",
          # model= "llama-3.1-8b-instant",
          messages=[
              {"role": "system", "content": self.response_prompt.format(context_summary=context_string)},
              {"role": "user", "content": [{"type": "text", "text": user_query}] + ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] if image_base64 else [])}
          ]
      )
      return response.choices[0].message.content
      # return self.response_prompt.format(context_summary=context_string)
    except Exception as e:
      print(f"An error occurred while calling the API: {e}")
      return None

  async def __call__(self, volumes, user_query=None, image_base64=None):
    await self.density_calculator(volumes)

    if user_query is not None:
      return await self.query_food_assistant(volumes, user_query, image_base64)
    else:
      return None

if __name__ == "__main__":
  # Note this was intended for testing purpose only before async shift for sadly fast api calls.
  from tavily import TavilyClient
  import os, base64
  from dotenv import load_dotenv
  from openai import OpenAI
  load_dotenv()

  BASE_DIR = os.path.dirname(__file__)
  input_image_path = BASE_DIR+"/uploads/test2.png"

  sample_volume = {
  "segments": [
    {
      "image_base64": "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",
      "segment_id": 1,
      "volume_ml": 58.41004580925604,
      "food_name": "green leafy vegetable",
      "confidence": 0.6,
      "major_uncertainties": [
        "Cannot tell if this is a specific type of leafy green (e.g., spinach, kale, or broccoli)",
        "Uncertain if it's cooked or raw, which affects calorie count"
      ],
      "most_important_question": "Is this item cooked or raw?",
      "ambiguity_flag": True
    },
    {
      "image_base64": "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",
      "segment_id": 2,
      "volume_ml": 85.86862210079667,
      "food_name": "chicken",
      "confidence": 0.6,
      "major_uncertainties": [
        "The food item appears to be a type of meat, but it's unclear if it's beef, pork, lamb, or chicken, which have different calorie counts.",
        "The cooking method (grilled, roasted, or fried) is also uncertain, which significantly impacts calorie count."
      ],
      "most_important_question": "Is this item chicken or red meat (beef, pork, lamb)?",
      "ambiguity_flag": True
    }
  ],
  "total_segments": 4
  }
  tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
  # print(os.getenv('TAVILY_API_KEY'))
  llm_client = OpenAI(
      api_key=os.getenv("GROQ_API_KEY"),
      base_url="https://api.groq.com/openai/v1"
  )

  parser = NutritionAgent(tavily_client, llm_client)
  print('-'*60)
  with open(input_image_path, 'rb') as image:
    image_b64 = base64.b64encode(image.read()).decode("utf-8")
    # print(image_b64)
    print(parser(sample_volume, "How much weight I will gain from it ?", image_b64))
  # with open('agents/final.json', 'w', encoding='utf-8') as f:
  #     json.dump(sample_volume, f, indent=2, ensure_ascii=False)

