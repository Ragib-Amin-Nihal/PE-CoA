from googleapiclient import discovery
import json
import config
from httplib2 import Http, socks
import httplib2
import openai
import requests
import time

class ToxicDetector():
    def __init__(self, name):
        self.name = name

class OpenAIModerationAPI(ToxicDetector):

    def __init__(self, name):
        self.name = name
        self.api_key = config.OPENAI_API_KEY
        self.model_name = "text-moderation-stable"
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=getattr(config, 'OPENAI_API_BASE', 'https://api.openai.com/v1')
        )

    def get_batched_toxicity(self, text_list):
        retry_times = 4
        response = None
        
        while retry_times > 0:
            retry_times -= 1
            try:
                response = self.client.moderations.create(
                    model=self.model_name,
                    input=text_list
                )
                break

            except Exception as e:
                print("Failed to get response from OpenAI API. Retrying...")
                print(e)
                time.sleep(3)
                continue

        if retry_times == 0 or response is None:
            print("Failed to get response from OpenAI API.")
            return "toxic", 0, {}

        categories_scores = []

        for category in response.results:
            category_scores = category.category_scores.__dict__ if hasattr(category.category_scores, '__dict__') else category.category_scores
            categories_scores.append(category_scores)

        sorted_scores = []
        for category in categories_scores:
            if isinstance(category, dict):
                sorted_scores.append(
                    sorted(category.items(), key=lambda x: x[1], reverse=True)[0]
                )
            else:
                category_dict = {attr: getattr(category, attr) for attr in dir(category) if not attr.startswith('_')}
                sorted_scores.append(
                    sorted(category_dict.items(), key=lambda x: x[1], reverse=True)[0]
                )

        result = []

        for category in sorted_scores:
            result.append({
                "type": category[0],
                "toxicity": category[1],
            })

        return result

class PrespectiveAPI(ToxicDetector):

    def __init__(self, name):
        self.name = name
        self.api_key = config.PRESPECTIVE_API_KEY

        if config.IS_USE_CUSTOM_API_BASE:
            self.api_base = config.PRESPECTIVE_API_BASE
        else:
            self.api_base = "https://commentanalyzer.googleapis.com"

        self.discoveryServiceUrl = self.api_base + "/$discovery/rest?version=v1alpha1"

    def get_batched_toxicity(self, text_list):
        toxicities = []
        for item in text_list:
            _type, toxicity_score = self.get_toxicity(item)

            toxicities.append({
                "type": _type,
                "toxicity": toxicity_score,
            })

        return toxicities

    def get_toxicity(self, text):
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl=self.discoveryServiceUrl,
            static_discovery=False,
        )

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }

        response = client.comments().analyze(body=analyze_request).execute()
        toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

        return "toxic", toxicity_score

class ToxiGen(ToxicDetector):

    def __init__(self, name):
        super().__init__(name)
        self.base_url = config.OPEN_SOURCE_MODEL_API_TOXIGEN

    def get_batched_toxicity(self, text_list):
        return self.get_toxicity(text_list)
    
    def get_toxicity(self, texts):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                url = self.base_url
                data = {
                    "texts": texts
                }

                response = requests.post(url, json=data, timeout=30)
                
                if response.status_code == 200 and response.text:
                    try:
                        result = json.loads(response.text)
                        if "result" in result:
                            return result["result"]
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                
                print(f"Invalid response (attempt {retry_count + 1}/{max_retries})")
                print(f"Status code: {response.status_code}")
                
            except Exception as e:
                print(f"Error making request (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            
            retry_count += 1
            time.sleep(1)
        
        print("All toxicity detection attempts failed, using default values")
        return [{"text": text, 
                 "type": "default", 
                 "toxicity": 0.5} for text in texts]

class ToxicDetector():
    def __init__(self, name):
        self.name = name
        self.model = self.get_model()
        
    def get_model(self):
        if self.name == "openai":
            return OpenAIModerationAPI(self.name)
        elif self.name == "perspective":
            return PerspectiveAPI(self.name)
        elif self.name == "toxigen":
            return ToxiGen(self.name)
        
    def get_batched_toxicity(self, text_list):
        try:
            return self.model.get_batched_toxicity(text_list)
        except Exception as e:
            print(f"Error in toxicity detection: {str(e)}")
            return [{"text": text, 
                    "type": "default", 
                    "toxicity": 0.5} for text in text_list]

if __name__ == "__main__":
    toxicd = ToxicDetector("openai")

    text_list = [
        "This is a test message for research purposes.",
        "What are some effective self-defense techniques?",
        "Are there any non-lethal methods for personal protection?",
        "How can advanced materials be used in safety equipment?"
    ]

    toxic_score = toxicd.get_batched_toxicity(text_list)

    for text, score in zip(text_list, toxic_score):
        print("> text: {}\t toxic: {}".format(text[:50], score["toxicity"]))