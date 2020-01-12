from pymongo import MongoClient
import pymssql

def get_mongo_conn():
    conn = MongoClient('localhost', 27017)
    db = conn['test_db']
    return db

def get_books_collection():
    db = get_mongo_conn()
    collection = db['top_books_collection']
    return collection

def get_genre_collection():
    db = get_mongo_conn()
    collection = db['top_books_genre_collection']
    return collection

def get_user_collection():
    db = get_mongo_conn()
    collection = db['top_user_books_collection']
    return collection

def get_library_collection():
    db = get_mongo_conn()
    collection = db['top_library_books_collections']
    return collection

def get_mssql_connection():

    conn = pymssql.connect(server='btdigitalsandbox.database.windows.net',
                           port='1433',
                           user='Dbadmin@btdigitalsandbox.database.windows.net',
                           password='BT!pw0915',
                           database='digitalsandbox')
    return conn



