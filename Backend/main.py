from flask import Flask,make_response, jsonify,request,json
from flask_cors import CORS
import os
import re
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)
cors = CORS(app)


@app.route('/identify',methods=['GET','POST'])
def identify():
    title = request.json['title']
    content = request.json['content']

    cmd = "python3 detect.py "+ '"'+str(title)+'"'
    os.system(str(cmd))
    f=open("output.txt","r")
    lines = f.readlines()
    lines[0]=re.sub("\n","",lines[0])
    print(lines[0])
    if lines[0]=="1":
        clickbait="yes"
    else:
        clickbait="no"

    newtitle=''
    if clickbait=="yes":
        cmd = "python3 summ.py "+ '"'+str(title)+'"'
        os.system(str(cmd))
        f=open("output.txt","r")
        lines = f.readlines()
        newtitle=lines[0]

    result={
        "clickbait": clickbait,
        "newtitle" : newtitle
    }
    print(result)
    return jsonify(result)

@app.route('/titleandcontent',methods=['GET','POST'])
def titleandcontent():
    url = request.json['url']
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)

    title1 = ''
    content1 = ''
    title = ['h1']
    content = ['p','h2','h3','h4','h5','h6']

    for t in text:
        if t.parent.name in title:
            title1 += '{} '.format(t)
        if t.parent.name in content:
            if content1=='':
                content1 += '{} '.format(t)

    result={
        "title": title1,
        "content": content1
    }

    print(result)
    return jsonify(result)



if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000,debug=True)
    app.run(debug=True)