from flask import Flask, jsonify, request
import pymongo

app = Flask(__name__)

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['mongodb_name']
collection_name = 'user_top_books'

db_cm = mng_db[collection_name]

@app.route('/top_books/<uid>', methods=['GET'])
def get_top_books(uid):
    cursor = db_cm.user_top_books.distinct(uid)
    return jsonify({'top_100_books':cursor})


if __name__ == '__main__':
    app.run(debug=True)

