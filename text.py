text=input("enter text:")
with open("sample.txt","w") as f:f.write(text)
with open("sample.txt") as f:print(f.read())
