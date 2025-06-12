from firebase_admin import credentials, db, initialize_app

# Initialize Firebase
cred = credentials.Certificate("autoclock-sriipuj-firebase.json")  # Ensure this file is in the same directory
initialize_app(cred, {
    "databaseURL": "https://autoclock-sriipuj-default-rtdb.asia-southeast1.firebasedatabase.app"  # Replace with your actual database URL
})

# Add staff information
staff_ref = db.reference("staff")  # Reference the "staff" node in Firebase
staff_data = {
    "staff1": {
        "name": "Afnan",
        "position": "teacher",
        "plate_number": "JFC2218"
    },
    "staff2": {
        "name": "Ahmad",
        "position": "counselor",
        "plate_number": "NAX7554"
    },
    "staff3": {
        "name": "Amin",
        "position": "teacher",
        "plate_number": "MDM1025"
    },
    "staff4": {
        "name": "Ajwad",
        "position": "librarian",
        "plate_number": "PFQ5217"
    }

}

# Populate the database with staff data
staff_ref.set(staff_data)
print("Staff information added successfully!")
