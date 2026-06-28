import sqlite3

db_name = "clashroyale.db"
print(f"Inspecting '{db_name}' structural schema...")

try:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 1. Fetch all table names inside your database file
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n--- FOUND TABLES ---")
    if not tables:
        print("[-] WARNING: This database file contains ZERO tables. It is completely empty!")
    else:
        for t in tables:
            table_name = t[0]
            print(f"[+] Found Table Name: '{table_name}'")
            
            # 2. Peek inside the table to see its column names
            try:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"    Available columns: {columns}")
            except Exception as col_err:
                print(f"    Could not read columns: {col_err}")
                
except Exception as e:
    print(f"Error reading database file: {e}")
finally:
    if 'conn' in locals():
        conn.close()
        