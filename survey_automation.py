from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

options = webdriver.ChromeOptions()
options.add_argument("--start-minimized")
options.add_experimental_option("detach", True) 

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.mcdolistens.com/")
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.ID, 'form1')))
def agree_click_button():
    try:
        button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "cr")))
        driver.execute_script("arguments[0].scrollIntoView();", button)
        time.sleep(5)
        button.click()
        print("Radio button clicked successfully!")
    except Exception as e:
        print("Error clicking the radio button:", e)
agree_click_button()
try:
    next_button = wait.until(EC.element_to_be_clickable((By.ID, "btnDisabled_tc")))
    next_button.click()
except Exception as e:
    print("Error clicking Let's Begin button:", e)

def multiple_choice(questions_xpath, answer_xpath):
    try:
        question = wait.until(EC.presence_of_element_located((By.XPATH, questions_xpath)))
        answer = question.find_element(By.XPATH, answer_xpath)
        answer.click()
    except Exception as e:
        print("Error selecting multiple choice:", e)
def answer_text(input_xpath, text):
    try:
        input_field = wait.until(EC.presence_of_element_located((By.XPATH, input_xpath)))
        input_field.send_keys(text)
    except Exception as e:
        print("Error entering text:", e)
answer_text('//input[@name="txtrestaurantnumber"]', '861')

time.sleep(5)

#driver.quit()
