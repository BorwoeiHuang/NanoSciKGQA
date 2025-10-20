import json
import re
from tqdm import tqdm
import bs4
import time
import re
from selenium import webdriver

def nanoletter_scrape(URL):
    
    browser = webdriver.Chrome()
    browser.get(URL)
    time.sleep(1)
    
    page = browser.page_source
    soup = bs4.BeautifulSoup(page)
    
    pattern = r'\">(.*?)</'

    try:
        title = soup.find("meta", attrs={'property': 'og:title'})["content"]
        abstract = soup.find("meta", attrs={'property': 'og:description'})["content"]
        keywords = soup.find("meta", attrs={'name': 'keywords'})["content"]
        datetime = soup.find("time")["datetime"]
        author = soup.find("span", attrs={"data-id": "article_author_info", 'property': 'author'})
        author = re.search(pattern, str(author)).group(1)
        subjects = soup.find_all("a", attrs={"class": "article__tags__link"})
        subjects = [item.replace("<sub>", "").replace(">", "").replace("</", "") for item in re.findall(pattern, str(subjects))]
        affiliation = soup.find("div", attrs={"class": "loa-info-affiliations-info"})
        affiliation = re.search(pattern, str(affiliation)).group(1)
        nanoletter_output_dict={"URL":URL, 
                               "Title":title,
                               "Date":datetime,
                               "Authors":author,
                               "Affiliation":affiliation,
                               "Keywords":keywords,
                               "Subjects":subjects,
                               "Abstract":abstract}
    except:
        nanoletter_output_dict = {}
        pass
        
        
    return nanoletter_output_dict

if __name__ == "__main__":
    
    next_DOI ="0001"
    for i in range(2,5):
        file_number = i
        out_path = f"nanoletter_abstracts/abstracts_{file_number}.jsonl"
        try:
            with open(out_path, 'r') as readfile:
                last_line = readfile.readlines()[-1]
            last_scraped_DOI = json.loads(last_line)["URL"][-5:]
        except:
            last_scraped_DOI = "0000"
        print(f"last scraped article DOI is: {file_number}c{last_scraped_DOI}")
        next_DOI = int(last_scraped_DOI)+1 
        with open(out_path, 'a') as outfile:
            for i in tqdm(range(next_DOI, 2300)):
                URL= f"https://pubs.acs.org/doi/10.1021/acs.nanolett.{file_number}c0" + f'{i:04}'
                nanoletter_output = nanoletter_scrape(URL)
                if len(nanoletter_output)!=0:
                    json.dump(nanoletter_output, outfile)
                    outfile.write('\n')
        
        