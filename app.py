from ChatbotProj import chat
from flask import Flask, render_template,request


app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def index():
    input = ""
    userText=""
    if request.method == "POST":
        
        userText = request.form['textInput']
        input = chat(userText)
        
        
    return render_template("index.html",inputs=input,user=userText)


@app.route('/faqs')
def faqs():
    return render_template("faqs.html")

@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(debug = False)