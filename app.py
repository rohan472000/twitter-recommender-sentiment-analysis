from flask import Flask, request, render_template

from social_twitter import recommend_users_for_user_final

# app = Flask(__name__)
app = Flask(__name__, template_folder='template')


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form['username']
    recommendations = recommend_users_for_user_final(username)  # Replace with the name of your function
    return render_template('recommend.html', username=username, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
