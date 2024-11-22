import sqlite3

def fetch_customer_data():
    connection = sqlite3.connect('crm_database.db')
    query = 'SELECT * FROM customers'
    return pd.read_sql_query(query, connection)
