import sqlite3

conn = sqlite3.connect("myra_memory.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM conversations")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
