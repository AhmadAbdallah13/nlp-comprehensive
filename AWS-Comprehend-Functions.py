import boto3
import json

comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')
text = ""

def DetectDominantLanguage(text):
    print('Calling DetectDominantLanguage')
    print(json.dumps(comprehend.detect_dominant_language(Text = text), sort_keys=True, indent=4))
    print("End of DetectDominantLanguage\n")

def DetectEntities(text):
    #You must specify the language of the input text.
    print('Calling DetectEntities')
    print(json.loads(json.dumps(comprehend.detect_entities(Text=text, LanguageCode='ar'), sort_keys=True, indent=4)))
    print('End of DetectEntities\n')

def DetectKeyPhrases(text):
    #determine the key noun phrases used in text, You must specify the language of the input text.
    print('Calling DetectKeyPhrases')
    print(json.loads(json.dumps(comprehend.detect_key_phrases(Text=text, LanguageCode='ar'), sort_keys=True, indent=4)))
    print('End of DetectKeyPhrases\n')

def DetectSentiment(text):
    #Detect sentiment, You must specify the language of the input text.
    print('Calling DetectSentiment')
    print(json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='ar'), sort_keys=True, indent=4))
    print('End of DetectSentiment\n')

def DoEverything(text, DetectSentimentt=True,
                     DetectKeyPhrasess=True, DetectEntitiess=True, 
                     DetectDominantLanguages=True):
    
    if DetectSentimentt:
        DetectSentiment(text)
    if DetectKeyPhrasess:
        DetectKeyPhrases(text)
    if DetectEntitiess:
        DetectEntities(text)
    if DetectDominantLanguages:
        DetectDominantLanguage(text)

def main():
    text = str(input("please enter a text to analyze:\n"))
    DoEverything(text)


if __name__ == "__main__":
    main()

