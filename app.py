from flask import Flask, render_template, request
from src.main import process_query

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        k = int(request.form['k'])
        answers_with_ids = process_query(query, k)
        return render_template('index.html', query=query, answers=answers_with_ids)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
