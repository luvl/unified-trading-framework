from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re, os

import urllib3
import certifi
from bs4 import BeautifulSoup

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd 
from googletrans import Translator

def prep_news(filename: str, save_as_name: str) -> None:
    with open(filename) as f:
        datas = json.load(f)

    news_list = []

    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )

    for data in datas:
        url = data['url']
        date_publish = data['datatime']
        title = data['title']

        response = http.request('GET', url)
        soup = BeautifulSoup(response.data, 'html.parser')
        news_div = soup.find('div', attrs={'id':'newscontent'})

        try:
            text = news_div.text
            text = re.sub(r"Các tập tin đính kèm(.*)\.pdf", "" , text)
        except:
            text = None

        tmp = {
            'title': title,
            'text': text,
            'date_publish': date_publish,
        }

        news_list.append(tmp)

    with open(save_as_name, "w") as fp:
        json.dump(news_list, fp, indent=2, ensure_ascii=False)

def eng_translate(filename: str, save_as_name: str) -> None:
    with open(filename, encoding='utf-8', errors='ignore') as json_data:
        news_json = json.load(json_data, strict=False)

    translator = Translator(
        service_urls = None, 
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 
        proxies = None, 
        timeout = None
    )

    for i in range(len(news_json)):
        news = news_json[i]['text']
        try:
            translated = translator.translate(news, src='vi', dest='en')
            news_json[i]['en_text'] = translated.text
        except:
            news_json[i]['en_text'] = None

    with open(save_as_name, "w") as fp:
        json.dump(news_json, fp, indent=2, ensure_ascii=False)

def update_eng_translate(save_as_name: str) -> None:
    with open(save_as_name, encoding='utf-8', errors='ignore') as json_data:
        news_json = json.load(json_data, strict=False)

    translator = Translator(
        service_urls = None, 
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 
        proxies = None, 
        timeout = None
    )

    for i in range(len(news_json)):
        if news_json[i]['en_text'] is None:
            news = news_json[i]['text']

            try:
                translated = translator.translate(news, src='vi', dest='en')
                print(translated.text)
                news_json[i]['en_text'] = translated.text
            except:
                news_json[i]['en_text'] = None
        else:
            continue

    with open(save_as_name, "w") as fp:
        json.dump(news_json, fp, indent=2, ensure_ascii=False)    

if __name__ == "__main__":
    if not os.path.exists('news.json'):
        prep_news('vinamilk.json', "news.json")

    if os.path.exists('translated_news.json'):
        update_eng_translate('translated_news.json') # continue to update the engtranslate since there is quote usage, better find a pretrained model to do this
    else:
        eng_translate('news.json', 'translated_news.json')

