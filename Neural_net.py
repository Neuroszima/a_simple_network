
import numpy
from matplotlib import pyplot as plt

'''
testowy projekt sieci neuronowych, który był przedstawiony w filmie

dane pochodzą z filmu na youtube: https://www.youtube.com/watch?v=LSr96IZQknc

są one posegregowane w następujący sposób:

wartości = tablica[
    wartość1[długość_płatka1, szerokość_płatka1, kolor_płatka1],
    wartość2[długość_płatka2, szerokość_płatka2, kolor_płatka2],
    ...,
]
'''


values_from_tutorial = [  # jest to poprostu ułożone w stylu: długość, szerokość, typ/klasa
    [3, 1.5, 1],
    [2, 1, 0],
    [4, 1.5, 1],
    [3, 1, 0],
    [3.5, .5, 1],
    [2, .5, 0],
    [5.5, 1, 1],
    [1, 1, 0],

]

unknown_point = [4.5, 1, "?"]  # nasz tajemniczy kwiatek


'''
pomyślmy teraz nad naszą siecią i jak ma ona wyglądać:

      O            kolor kwiatka (rezultat)
     / \ + b      waga1    waga2   + odchylenie stałe(miara fałszu naszych wyników)
    O   O       szerokość, długość kwiatka (wejście)

'''

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

print(w1, w2, b)
print(values_from_tutorial)

'''
definiujemy funcję dzięki której będziemy wiedzieć jak blisko/daleko
jesteśmy od pożądanego wyniku

innymi słowy będzie to reprezentacja "przekonania" naszej sieci o tym,
do jakiej klasy należy dany punkt

taką funkcją będzie "sigmoid"

funkcja ta kondensuje nam wartości sieci neuronowej do takiej której dziedzina
mieści się w zakresie (0, 1), odpowiadająca w pewnym stopniu "przekonaniu"
sieci neuronowej, do przypożądkowania pewnej danej do klasy 0 lub 1
'''


def sigmoid(x):
    return 1/(1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

'''
Wyświetlmy naszą funkcję; 
jako X damy tutaj tablicę T wartości X przykładowo 10 punktów w przestrzeni,
a nastpęnie wyświetlimy je na wykresie dzięki matplotlib
'''

T = numpy.linspace(-5, 5, 100)
# print(T)
Y = sigmoid(T)
# print(Y)

plt.plot(T, Y)
plt.show()

Y = sigmoid_derivative(T)

plt.plot(T, Y)
plt.show()

'''
Spójrzmy sobie na nasze dane
'''

for point in values_from_tutorial:
    x = point[0]
    y = point[1]
    color = 'red'
    if point[2] == 0:
        color = 'blue'
    else:
        pass
    plt.scatter(x, y, c=color)

plt.show()

'''
pora na pętlę uczącą,

generalnie bierzemy punkt, puszczamy go przez sieć, patrzymy czy jest dobrze czy źle

jak jest źle, bierzemy wynik sieci, porównujemy z tym co miało być, odejmujemy od siebie

to będzie nasz "koszt obliczeniowy"

użyjemy tego kosztu by poprawić nieco przewidywanie, poprzez modyfikację wag oraz stałego odchylenia,
 
i tak w kółko dopóki nie zdecydujemy czy jesteśmy zadowoleni z wyników,
albo dopóki ilość powtórzeń pętli się nie wyczerpie
'''
iterations = 100000  # ilość powtórzeń pętli

learning_rate = 0.1  # szybkość uczenia się sieci, wyjaśniona niżej

cost_table = []  # będziemy zbierać tu dane o postępach nauki naszej sieci, wyjaśnione niżej

for _ in range(iterations):
    # wybieramy punkt danych losowo
    point_index = numpy.random.randint(len(values_from_tutorial))
    point = values_from_tutorial[point_index]
    # print(point, point_index)

    # patrzymy, jak zachowuje się - jak "myśli" nasza sieć:
    net_output = point[0]*w1 + point[1]*w2 + b

    '''
    dokonujemy predykcji; wartości z sieci zostają przetworzone na takie, które mogą mieć
    wartość z zakresu (0, 1)
    '''

    prediction = sigmoid(net_output)

    '''
    musimy następnie policzyć koszt obliczenia - miarę "pomyłki"
    naszej sieci od wartości oczekiwanej
    robimy to w stylu podobnym do obliczania metody najmniejszych kwadratow
    '''

    target = point[2]
    cost = numpy.square(prediction - target)

    '''
    wyznaczamy różniczkę funkcji kosztu naszego przewidywania
    robimy to po to by wiedzieć czy i jak daleko jesteśmy od minimum tej funkcji
    dodatkowo znak różniczki pomoże nam odnaleźć po tórej stronie minimum się znajdujemy
    czyli w którą stronę mamy iść (w lewo czy prawo)
    '''

    dcost_dprediction = 2*(prediction - target)

    '''
    dostaliśmy różniczkę kosztu funkcji, teraz wiemy w którą stronę mamy iść
    (czy mamy znak "-" czy "+"), oraz jak daleko jesteśmy
    trzeba teraz odpowiednio zmodyfikować nasze wagi oraz miarę odchylenia stałego
    (czyli tzw. "bias") tak, by następnym razem lepiej przewidzieć klasę dla danej wartości
    
    robimy to poprzez kolejne różniczkowanie, tym razem funkcji wewnętrznej, ponieważ mamy
    3 zmienne które musimy poprawić:
        wagę 1 (w1)
        wagę 2 (w2)
        odchylenie stałe (b, bias, miara fałszu)
        
    nasze funkcje wyglądały tak: 
        net_output = point[0]*w1 + point[1]*w2 + b
        prediction = sigmoid(net_output)
        target = point[2]
        cost = numpy.square(prediction - target)
    
    a więc:
    '''

    dprediction_dnetoutput = sigmoid_derivative(net_output)

    dnetoutput_dw1 = point[0]
    dnetoutput_dw2 = point[1]
    dnetoutput_db = 1

    dcost_dw1 = dcost_dprediction * dprediction_dnetoutput * dnetoutput_dw1
    dcost_dw2 = dcost_dprediction * dprediction_dnetoutput * dnetoutput_dw2
    dcost_db = dcost_dprediction * dprediction_dnetoutput * dnetoutput_db

    '''
    mamy różniczki względem każdego parametru, a więc teraz możemy wyznaczyć jak te poszczególne wagi
    mają się zmienić by poprawić nasze przewidywania w następnych iteracjach
    
    zazwyczaj dajemy pewien parametr szybkości uczenia, robimy to po to by nie "przestrzelić" minimum
    i w rezultacie nie wyminąć go i oddalać się od niego
    '''

    w1 -= learning_rate*dcost_dw1
    w2 -= learning_rate*dcost_dw2
    b -= learning_rate*dcost_db

    '''
    moglibyśmy tu wystartować, można jednak stworzyć pewien ciekawy wykres, np. krzywą uczenia
    
    zbierzmy dane co np. 5 iteracji, umieśćmy je w tablicy i przedstawmy je poza pętlą treningową na wykresie
    
    zbieramy całkowity błąd z przewidywania wszystkich punktów
    '''

    if _ % 20 == 0:  # tutaj jest co 20 iteracji
        # print(_)
        total_cost = 0
        for p in values_from_tutorial:
            net_out = p[0] * w1 + p[1] * w2 + b
            pred = sigmoid(net_out)
            cst = numpy.square(pred - p[2])
            # print(p, pred, cst)
            total_cost += cst
        cost_table.append(total_cost)


'''
wreszcie po zakończonym treningu, możemy spojrzeć na wyniki i sprawdzić jak nasza sieć się sprawdziła

wykreślamy całkowity błąd a następnie patrzymy jakie przewidywania są dla wybranych punktów
'''

plt.plot(cost_table)
plt.show()

print('final coefficients:', w1, w2, b)

for i in range(len(values_from_tutorial)):
    point = values_from_tutorial[i]
    net_output = point[0]*w1 + point[1]*w2 + b
    pred = sigmoid(net_output)
    print(point, pred)


# a jak tam nasz tajemniczy kwiatek?
unknown_point_pred = unknown_point[0]*w1 + unknown_point[1]*w2 + b
unknown_point_pred = sigmoid(unknown_point_pred)
print('\n and what about mystery flower?')
print(unknown_point, unknown_point_pred)

color_un = 'red'
if unknown_point_pred < 0.5:
    color_un = 'blue'

print("it's : " + color)
plt.scatter(unknown_point[0], unknown_point[1], c=color)
for point in values_from_tutorial:
    x = point[0]
    y = point[1]
    color = 'red'
    if point[2] == 0:
        color = 'blue'
    else:
        pass
    plt.scatter(x, y, c=color)

plt.show()

