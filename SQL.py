import sqlite3
import sqlite3
import pandas as pd

def create_database(database_name): 
    try:
        conn = sqlite3.connect(f"storage/{database_name}.db")
        cur = conn.cursor()
        
    except sqlite3.Error as e:
        print("Database error:", e)
    
    finally:
        if conn:
            conn.close()
        print("Connection closed")


def create_table(database_name,table_name):
    conn = sqlite3.connect(f"storage/{database_name}.db")
    cur = conn.cursor()
    
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            filepath TEXT NOT NULL UNIQUE,
            phash TEXT,
            whash TEXT,
            is_duplicate BOOLEAN,
            embedding BLOB
        )
        """)
    
    conn.commit()
    print("Table created successfully")


def view_table(database_name,table_name) : 
    try:
        conn = sqlite3.connect(f"storage/{database_name}.db")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(df.to_string(index=False))
    except sqlite3.Error as e:
        print("Database error:", e)
    
    finally:
        if conn:
            conn.close()
        print("Connection closed")



def list_database(database_name) :
    try:
        conn = sqlite3.connect(f"storage/{database_name}.db")
        cur = conn.cursor()
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        
        conn.close()
        
        for table in tables:
            print(table[0])
    except sqlite3.Error as e:
        print("Database error:", e)
    
    finally:
        if conn:
            conn.close()
        print("Connection closed")    


def add_entry(database_name,table_name,filename,filepath):
    try:
        conn = sqlite3.connect(f"storage/{database_name}.db")
        cur = conn.cursor()

        cur.execute(f"""
            INSERT INTO {table_name} (filename, filepath)
            VALUES (?, ?)
        """, (filename, filepath))

        conn.commit()
        row_id = cur.lastrowid
        print(f"Inserted basic entry with id = {row_id}")
        return row_id

    except sqlite3.Error as e:
        print("Database error:", e)

    finally:
        conn.close()



def update_entry(database_name, table_name, row_id,
                 phash=None, whash=None,
                 is_duplicate=None, embedding=None):

    try:
        conn = sqlite3.connect(f"storage/{database_name}.db")
        cur = conn.cursor()

        fields = []
        values = []

        if phash is not None:
            fields.append("phash = ?")
            values.append(phash)

        if whash is not None:
            fields.append("whash = ?")
            values.append(whash)

        if is_duplicate is not None:
            fields.append("is_duplicate = ?")
            values.append(is_duplicate)

        if embedding is not None:
            fields.append("embedding = ?")
            values.append(embedding)

        if not fields:
            print("Nothing to update")
            return

        values.append(row_id)

        query = f"""
            UPDATE {table_name}
            SET {", ".join(fields)}
            WHERE id = ?
        """

        cur.execute(query, tuple(values))
        conn.commit()
        print(f"Updated entry id = {row_id}")

    except sqlite3.Error as e:
        print("Database error:", e)

    finally:
        conn.close()
