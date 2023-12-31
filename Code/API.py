from flask import Flask, request, jsonify
from recommendation import keywords_recommendation

app = Flask(__name__)

# ... (Các phần khác của Flask app)

@app.route('/keywords_recommendation', methods=['POST'])
def recommendation():
    data = request.get_json()
    query_keywords = data.get('keywords', [])
    top_n = data.get('top_n', 5)
    recommended_courses = keywords_recommendation(query_keywords, top_n)
    return jsonify(recommended_courses)

if __name__ == '__main__':
    app.run(debug=True)
