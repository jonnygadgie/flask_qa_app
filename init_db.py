import sqlite3

connection = sqlite3.connect('database3.db')

with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO posts (title, years, datas, parameters, finalparameters ) VALUES (?, ?, ?, ?, ?)",
            ('First Post', '1990, 1991, 1992, 1993', '1, 2, 3, 4', '2, 1, 1, 1, 1, 2, 2, 2, 2', '2, 1, 1, 1, 1, 2, 2, 2, 2')
            )

cur.execute("INSERT INTO posts (title, years, datas, parameters, finalparameters) VALUES (?, ?, ?, ?, ?)",
            ('Second Post', '1994, 1995, 1996, 1997', '5, 6, 7, 8', '2, 1, 1, 1, 1, 2, 2, 2, 2', '2, 1, 1, 1, 1, 2, 2, 2, 2')
            )

connection.commit()
connection.close()