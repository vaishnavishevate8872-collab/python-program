expenses=[]
while True:
    c=input("1 Add 2 View 3 Exit:")
    if c=="1":
        expenses.append(float(input("Amount:")))
    elif c=="2":
        print(expenses,"Total=",sum(expenses))
    else: break