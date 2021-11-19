from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd

main_url = "https://03online.com"

#Covid section
covid_url_part1 = "https://03online.com/news/covid_19/"
covid_url_part2 = "-0-68"
allergist_part1 = "https://03online.com/news/allergolog/"
allergist_part2 = "-0-23"
dermatologist_part1 = "https://03online.com/news/dermatolog/"
dermatologist_part2 = "-0-18"
cardiologist_part1 = "https://03online.com/news/kardiolog/"
cardiologist_part2 = "-0-7"
pulmonologist_part1 = "https://03online.com/news/pulmonolog/"
pulmonologist_part2 = "-0-27"

part1 = dermatologist_part1
part2 = dermatologist_part2
question_answers = []
question_texts = []

for page_id in range(2,252):
    Url = "".join((part1, str(page_id), part2))
    page = urlopen(Url)
    soup_object = BeautifulSoup(page, "html.parser")
    questions = []
    for question in soup_object.find_all("div", class_="question-short-block"):
        questions.append(main_url + question.find("div", class_="header").find("a").get('href'))
    
    
    for object_ques in questions:
        question_page = urlopen(object_ques)
        soup_question_page = BeautifulSoup(question_page, "html.parser")
        question_texts.append(soup_question_page.find("div", class_="text").text)
        #For collecting Doctors' answers
        #if(soup_question_page.find("div", class_="answer-block doctor-block")) is not None:
        #    question_answers.append(soup_question_page.find("div", class_="answer-block doctor-block").find(class_="content").text)
        #else:
        #    question_answers.append("")
    print(page_id)
        
data = {'X':question_texts}
data["Y"] = 4
df = pd.DataFrame(data)
df.index += 1
df.to_csv("dermatologist.csv")

