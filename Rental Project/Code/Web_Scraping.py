#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


# In[2]:


# Start the timer
start_time = time.time()


# In[3]:


# Connect to the webpage
driver = webdriver.Chrome()

# URL of the website to scrape
url = 'https://www.99.co/singapore/rent/rooms'  

# Visit the URL
driver.get(url)

# Wait for the page to fully load (adjust timeout as needed)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, '_17qrb')))


# In[4]:


# Lists to store scraped data
address_list = []
address_list_2 = []
nearest_mrt_list = []
time_to_nearest_mrt_list = []
unit_type_list = []
room_type_list = []
room_size_list = []
room_size_list_2 = []
price_list = []
status_list = []
updated_time_list = []
link_list = []


# In[5]:


# Loop to scrape data from multiple pages

pagenum = 1

while True:
    try:
        # Get the page source and parse it with BeautifulSoup
        my_page = driver.page_source
        my_html = BeautifulSoup(my_page, "html.parser")

        # Find all containers
        containers = my_html.findAll('div', {'class': '_17qrb _1QP8Y'})

        # Extract data from each container
        for i in range(len(containers)):

            try:
                address = containers[i].find('h2').find('a')['title']
            except:
                address = 'unknown'

            try:
                address_2 = containers[i].findAll('li',{'class':'_35Ugp'})[0].text
            except:
                address_2 = 'unknown'

            try:
                nearest_mrt = containers[i].findAll('p',{'class':'dniCg _1RVkE _2rhE- _1c-pJ'})[0].text
            except:
                nearest_mrt = 'unknown'

            try:
                time_to_nearest_mrt = containers[i].findAll('p',{'class':'dniCg _1RVkE _2rhE- _1c-pJ'})[1].text
            except:
                time_to_nearest_mrt = 'unknown'

            try:
                unit_type = containers[i].findAll('span',{'class':'_3hW-E'})[0].text
            except:
                unit_type = 'unknown'

            try:
                room_type = containers[i].findAll('li',{'class':'_1x-U1'})[0].text
            except:
                room_type = 'unknown'

            try:
                room_size = containers[i].findAll('li',{'class':'_1x-U1'})[1].text
            except:
                room_size = 'unknown'

            try:
                room_size_2 = containers[i].findAll('li',{'class':'_1x-U1'})[2].text
            except:
                room_size_2 = 'unknown'

            try:
                price = containers[i].findAll('ul',{'class':'_3XjHl'})[0].text
            except:
                price = 'unknown'

            try:
                status = containers[i].findAll('div',{'class':'TBTkb _36xav'})[0].text
            except:
                status = 'unknown'

            try:
                updated_time = containers[i].findAll('p',{'class':'_2y86Q _1dAt8 _2rhE-'})[0].text
            except:
                updated_time = 'unknown'

            try:
                link = containers[i].find('h2').find('a')['href']
            except:
                link = 'unknown'

            # Append data to lists
            address_list.append(address)
            address_list_2.append(address_2)
            nearest_mrt_list.append(nearest_mrt)
            time_to_nearest_mrt_list.append(time_to_nearest_mrt)
            unit_type_list.append(unit_type)
            room_type_list.append(room_type)
            room_size_list.append(room_size)
            room_size_list_2.append(room_size_2)
            price_list.append(price)
            status_list.append(status)
            updated_time_list.append(updated_time)
            link_list.append(link)

        print('Page:', pagenum)
        pagenum += 1

        # Check if there's a "Next" button and click it
        try:
            next_button = driver.find_element(By.LINK_TEXT, "Next")
            next_button.click()
            print(len(address_list))
            # Add a delay between page loads
            time.sleep(2)  
        except:
            print('End of pages')
            break
            
    except Exception as e:
        # Handle exceptions for the entire page scraping process
        print(f"Error while scraping page: {str(e)}")


# In[6]:


# Display the number of extracted data
number_of_row = len(address_list)
print(f"{number_of_row} rows of data are extracted.")


# In[7]:


# Clean up and close the web driver
driver.quit()


# In[8]:


# Create a DataFrame from the scraped data

scraped_data = pd.DataFrame({'address': pd.Series(address_list),
                             'address_2': pd.Series(address_list_2),
                             'nearest_mrt': pd.Series(nearest_mrt_list),
                             'time_to_nearest_mrt': pd.Series(time_to_nearest_mrt_list),
                             'price': pd.Series(price_list),
                             'unit_type': pd.Series(unit_type_list),
                             'room_type': pd.Series(room_type_list),
                             'room_size': pd.Series(room_size_list),
                             'room_size_2': pd.Series(room_size_list_2),
                             'status': pd.Series(status_list),
                             'updated_time': pd.Series(updated_time_list),
                             'link': pd.Series(link_list)})

# Save the newly scraped data to the CSV file
scraped_data.to_csv('property_listings.csv', index=False)


# In[9]:


# Display the total number of data
number_of_data = len(scraped_data)
print(f"There are {number_of_data} of data in total.")


# In[10]:


# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Scraping took {elapsed_time} seconds.")

