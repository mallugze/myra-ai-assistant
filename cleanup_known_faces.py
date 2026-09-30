import os
import shutil
import sqlite3

DB_PATH = "myra_memory.db"
BACKUP_PATH = "myra_memory.db.bak"
KNOWN_FACES_DIR = "known_faces"

print("--- [CLEANUP] Starting cleanup of known faces and memory except Mallu ---")

# 1. Backup the database
if os.path.exists(DB_PATH):
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"--> Database backed up to {BACKUP_PATH}")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 2. Clean known_faces table (Keep ONLY Mallu)
cursor.execute("SELECT id, name FROM known_faces WHERE LOWER(TRIM(name)) != 'mallu'")
removed_faces = cursor.fetchall()
print(f"--> Removing {len(removed_faces)} face record(s) from known_faces table:")
for r in removed_faces:
    print(f"    - ID {r['id']}: {r['name']}")

cursor.execute("DELETE FROM known_faces WHERE LOWER(TRIM(name)) != 'mallu'")

# 3. Clean known_people table (Keep ONLY Mallu)
cursor.execute("SELECT id, name, relationship FROM known_people WHERE LOWER(TRIM(name)) != 'mallu'")
removed_people = cursor.fetchall()
print(f"--> Removing {len(removed_people)} people record(s) from known_people table:")
for r in removed_people:
    print(f"    - ID {r['id']}: {r['name']} ({r['relationship']})")

cursor.execute("DELETE FROM known_people WHERE LOWER(TRIM(name)) != 'mallu'")

# 4. Clean conversation memory of non-Mallu interactions
other_names = ["bhajshi", "sharma", "manish", "testrahul", "chaitanya"]
total_convs_removed = 0
for name in other_names:
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE LOWER(content) LIKE ?", (f"%{name}%",))
    count = cursor.fetchone()[0]
    if count > 0:
        cursor.execute("DELETE FROM conversations WHERE LOWER(content) LIKE ?", (f"%{name}%",))
        total_convs_removed += count
        print(f"    - Removed {count} conversation log(s) mentioning '{name}'")

print(f"--> Total conversation logs removed from memory: {total_convs_removed}")

conn.commit()

# Rebuild and optimize database
cursor.execute("VACUUM")
conn.close()

# 5. Clean known_faces filesystem directory (Keep ONLY mallu.jpg)
if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        file_path = os.path.join(KNOWN_FACES_DIR, filename)
        file_lower = filename.lower()
        if file_lower in ["mallu.jpg", "mallu.jpeg", "mallu.png"]:
            print(f"--> Preserving: {file_path}")
        else:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"--> Deleted file from known_faces/: {filename}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"--> Deleted directory from known_faces/: {filename}")
            except Exception as e:
                print(f"--> Error deleting {file_path}: {e}")

print("\n--- [VERIFICATION] Checking state after cleanup ---")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT id, name FROM known_faces")
faces = cursor.fetchall()
print(f"known_faces remaining ({len(faces)}): {[dict(r) for r in faces]}")

cursor.execute("SELECT id, name, relationship FROM known_people")
people = cursor.fetchall()
print(f"known_people remaining ({len(people)}): {[dict(r) for r in people]}")

cursor.execute("SELECT key, value FROM personality_memory")
p_mem = cursor.fetchall()
print(f"personality_memory ({len(p_mem)}): {[dict(r) for r in p_mem]}")

for name in other_names:
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE LOWER(content) LIKE ?", (f"%{name}%",))
    print(f"conversations mentioning '{name}': {cursor.fetchone()[0]}")

conn.close()

files_remaining = os.listdir(KNOWN_FACES_DIR) if os.path.exists(KNOWN_FACES_DIR) else []
print(f"known_faces directory contents: {files_remaining}")

print("\n--- [CLEANUP COMPLETE] Only Mallu remains in database, memory, and filesystem! ---")
