books=[]
while True:
    c=input("1 Add 2 View 3 exit:")
    if c=="1": books.append(input("Books:"))
    elif c=="2": print(books)
    else: break