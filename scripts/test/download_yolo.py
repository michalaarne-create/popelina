import math
a = float(input("Podaj a"))
b = float(input("Podaj b"))

if a >= 0 and b >= 0:
    print("liczby są poprawne")
    c = math.sqrt(a*a + b*b)

    
    print("c wynosi", c)
elif a < 0 or b < 0:
    print("liczby są niepoprawne")

    


