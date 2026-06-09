from openai import OpenAI
import json

class DialogueAgent:
    def __init__(self, llm_client, input_callback=None):
        self.llm_client = llm_client
        # self.input_callback = input_callback 

    def confirm_analysis(self, volumes) -> list[dict] | bool:
        
        print("\n" + "="*70)
        print("FOOD CLARIFICATION")
        print("="*70 + "\n")
        
        # STEP 1: Collect all questions from all segments
        need_justifications = []
        for segment in volumes['segments']:
            
            # Add question if segment needs clarification
            most_important_question = segment['most_important_question'].strip()
            needs_clarification = (
                ((segment['ambiguity_flag'] and segment['confidence'] < 0.8) or (segment['confidence'] < 0.7)) and most_important_question
                # or len(major_uncertainties) > 0
            )
            if needs_clarification:
                # justification = input(most_important_question)
                need_justifications.append({'segment_id':segment['segment_id'], 'food_name':segment['food_name'], 'major_uncertainties': '\n'.join(segment['major_uncertainties']), 'question_asked':most_important_question})

        if len(need_justifications) > 0:
            return need_justifications
        else:
            return False

    async def _parse_bulk_answers(self, justifications, volumes):
        """
        Parse user justifications and identify food name corrections.

        Args:
            justifications (list[dict]): [
                {
                    "segment_id": "...",
                    "food_name": "...",
                    "most_important_question": "...",
                    "justification": "..."
                }
            ]

        Returns:
            list[dict]: Updated justifications list with corrected food_name values.
        """

        prompt = f"""
You are given a list of food segments and user justifications.

DATA:
{json.dumps(justifications, ensure_ascii=False, indent=2)}

TASK:
Determine whether the user's justification indicates that the current food_name is incorrect.

For each segment:
- If the user clearly specifies a different food item, return the corrected food name.
- If the justification does not imply a food name change, return null.
- Do not guess.
- Return ONLY valid JSON.

OUTPUT FORMAT:
{{
    "answers": {{
        "<segment_id>": "<new_food_name>" | null
    }}
}}
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a food annotation assistant. "
                            "Extract only explicit food name corrections. "
                            "Always return valid JSON."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)
            answers = result.get("answers", {})
            print(answers)
            for segment in volumes['segments']:
                new_food_name = answers.get(str(segment["segment_id"]))

                if new_food_name:
                    segment["food_name"] = new_food_name.strip()

        except Exception as e:
            print(f"Answer parsing failed: {e}")

    def _call_synthesizer(self, vlm_name, uncertainties, questions, user_answer, volume):
        """Synthesize final food name with user clarifications"""
        
        prompt = f"""Create detailed food name from user input.

VLM IDENTIFIED: {vlm_name}
VOLUME: {volume:.3f} litres

UNCERTAINTIES:
{chr(10).join(uncertainties) if uncertainties else "None"}

QUESTION ASKED:
{questions[0] if questions else "None"}

USER ANSWERED:
{user_answer}

TASK:
Create a detailed, specific food name incorporating user's clarification.

RULES:
- If user clarified food identity → update food name
- If user mentioned specifics → include them
- Be descriptive but concise

EXAMPLES:
User: "paneer not tofu" → "Paneer Curry"
User: "plain rice no oil" → "Plain Steamed Rice (No Oil)"
User: "it's grilled" → "Grilled [Food]"

OUTPUT JSON:
{{
  "food_name": "detailed name",
  "clarifications": {{"key": "value"}}
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a food identification assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            data = json.loads(response.choices[0].message.content)
            
            return {
                "food_name": data.get('food_name', vlm_name),
                "volume_litres": volume,
                "clarifications": data.get('clarifications', {}),
                "user_response": user_answer,
                "questions_asked": questions
            }
            
        except Exception as e:
            print(f"Synthesizer failed: {e}")
            return {
                "food_name": f"{vlm_name} [{user_answer[:30]}]",
                "volume_litres": volume,
                "clarifications": {},
                "user_response": user_answer,
                "questions_asked": questions
            }

    def _refine_with_suggestions(self, results, suggestions):
        """Refine results with additional suggestions"""
        
        prompt = f"""Update food names based on additional user suggestions.

CURRENT:
{json.dumps([{
    'id': r['segment_id'],
    'name': r['final_food_name'],
    'volume': r['volume_litres']
} for r in results], indent=2)}

USER SUGGESTIONS:
{suggestions}

Update relevant items based on suggestions. Keep others unchanged.

OUTPUT JSON:
{{
  "updates": [
    {{"segment_id": 1, "new_name": "updated name", "changed": true}},
    {{"segment_id": 2, "changed": false}}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            data = json.loads(response.choices[0].message.content)
            updates = data.get('updates', [])
            
            for update in updates:
                if update.get('changed'):
                    seg_id = update['segment_id']
                    for r in results:
                        if r['segment_id'] == seg_id:
                            r['final_food_name'] = update['new_name']
                            r['additional_suggestions'] = suggestions
            
            return results
            
        except Exception as e:
            print(f"Refinement failed: {e}")
            for r in results:
                r['additional_suggestions'] = suggestions
            return results
