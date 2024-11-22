from bs4 import BeautifulSoup
import requests

def scrape_product_prices(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    prices = [price.text for price in soup.find_all(class_='product-price')]
    return prices
