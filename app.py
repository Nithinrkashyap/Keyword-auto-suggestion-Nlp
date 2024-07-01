from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter

with open('autocorrect book.txt', 'r', encoding='utf-8') as f:
    data=f.read()
    data=data.lower()
    words = re.findall('\w+', data)

app = Flask(__name__)

V=set(words)
words_freq_dict=Counter(words)
probs={}
total_counts_words=sum(words_freq_dict.values())


for key in words_freq_dict.keys():
    probs[key]=words_freq_dict[key]/total_counts_words



    
    
@app.route('/')
def index():
    #* we have return because when i do get request it expects response
    return render_template('index.html',suggests=None,keyword=None)


@app.route('/suggest',methods=['POST','GET'])
def suggest():
    keyword = request.form['keyword'].lower()
    if keyword:
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, keyword) for v in words_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df.columns = ['Word', 'Prob']
        df['Similarity'] = similarities
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False).head(3)
        suggestions_list = suggestions.to_dict('records') 
        return render_template('index.html', suggestions=suggestions_list,keyword=keyword)

        
 



if __name__ == '__main__':
    app.run(debug=True)
