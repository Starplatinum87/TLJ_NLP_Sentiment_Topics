import pandas as pd
import numpy as np

import re 

from selenium import webdriver
from bs4 import BeautifulSoup
import time

import pickle

# Open IMDbPro Log in page 
driver = webdriver.Chrome('/Applications/chromedriver')
driver.get('https://www.imdb.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.imdb.com%2Fap-signin-handler&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=imdb_us&openid.mode=checkid_setup&siteState=eyJvcGVuaWQuYXNzb2NfaGFuZGxlIjoiaW1kYl91cyIsInJlZGlyZWN0VG8iOiJodHRwczovL3d3dy5pbWRiLmNvbS8_cmVmXz1sb2dpbiJ9&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&tag=imdbtag_reg-20')

# Grab password and login from file
login = open('<file_location_with_login_info', 'r').read()
login = login.split('\n')[0:2]

# Enter email and password on login page
driver.find_element_by_xpath('//*[@id="ap_email"]').send_keys(login[0]) # Enter your own login info
driver.find_element_by_xpath('//*[@id="ap_password"]').send_keys(login[1]) # Enter your own login info
driver.find_element_by_xpath('//*[@id="signInSubmit"]').click()

# Go to reviews page
driver.get('https://www.imdb.com/title/tt2527336/reviews?ref_=tt_ov_rt')

# Get page into BeautifulSoup object 
html = driver.page_source
soup = BeautifulSoup(html)

# Calculate number of scrolls needed
header = soup.find_all('div', class_='header')
review_num = re.findall(r'<span>(.+)\sReviews', str(header)) 
review_num = int(review_num[0].replace(',', ''))
loads = (review_num//25) + 6 # Each Load More adds 25 reviews

# Load new pages until all pages are present
for i in range(loads):
    try:
        driver.find_element_by_xpath('//*[@id="load-more-trigger"]').click()
        time.sleep(10)
    except:
        continue

# Get page into BeautifulSoup object 
html = driver.page_source
soup = BeautifulSoup(html)
imdb_reviews_raw = soup.find_all('div', {'class' : re.compile("(text show-more__control|text show-more__control clickable)")})
imdb_review_list = list(imdb_reviews_raw)


# Put all reviews text into list of reviews
TAG_RE = re.compile(r'(<[^>]+>|\n)')
def remove_tags(text):
    return TAG_RE.sub("", text)

clean_imdb_reviews = []
for i in imdb_review_list:
    review = remove_tags(str(i))
    clean_imdb_reviews.append(review)


# Get review containers soup object
imdb_review_containers = soup.find_all('div', class_="review-container")

# Create list of actual scores on page
imdb_scores_raw = soup.find_all('div', class_="ipl-ratings-bar") # Find all reviews with any rating
imdb_scores_raw_list = list(imdb_scores_raw)

# Create list of scores that were left with reviews, since a number of reviews did not have scores
found_scores = re.findall(r"<span>(\d+)<\/span>", str(imdb_scores_raw_list))

# Create list of scores with actual scores as numbers and missing scores as NaN
true_scores = []
count = 0 # Keep track of indices of found_scores list 
for i in found_scores:
    if i == True:
        true_scores.append(found_scores[count])
        count+=1
    else:
        true_scores.append(np.nan)

# Create series from each list
review_series = pd.Series(clean_imdb_reviews)
scores_series = pd.Series(true_scores)
TLJ_Reviews = pd.DataFrame({'Reviews': review_series, 'Scores': scores_series})

# Pickle results
write_data = TLJ_Reviews
pickle.dump(write_data, open('IMDb_TLJ_Reviews.pkl', 'wb'))

# Save results to csv file
TLJ_Reviews.to_csv('IMDb_TLJ_Reviews.csv')


