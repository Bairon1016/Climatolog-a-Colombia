import pymssql

def get_connection():
    conn = pymssql.connect(
        server='ClimaColombiaDB.mssql.somee.com',
        user='OscarTorres24_SQLLogin_1',
        password='et31h8v8pl',
        database='ClimaColombiaDB'
    )
    return conn