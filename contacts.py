contacts={}
while True:
    c=input("1 Add 2 View 3 Exit:")
    if c=="1":
        contacts[input("Name:")]=input("phone:")
    elif c=="2":
     print(contacts)
    else: break