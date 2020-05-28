<h3 align="center">
<p>A unified Machine Learning / Deep Learning framework for security trading and predicting traders’ strategies
</h3>

## Installation

Clone the repo and install dependencise, please active virtual environment if possible

```bash
git clone https://github.com/luvl/unified-trading-framework
cd unified-trading-framework
pip install -r requirements.txt
```

## Scrape the news

Srape vinamilk news, output vinamilk.json file 

```bash
scrapy crawl cafef-vnm-news -o vinamilk.json
```

## Start training

```bash
python main.py
```

## Deactive working process

```bash
pkill -9 java
```
