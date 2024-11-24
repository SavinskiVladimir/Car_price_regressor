import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import names
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

#функция вывода большого списка на экран
def out(a):
    count = 0
    for i in sorted(a):
        if count != 7:
            print(i, end=', ')
            count += 1
        else:
            print()
            count = 0
            print(i, end=', ')
    print()

# получаем данные для обучения регрессора из файла car_data.csv
data = pd.read_csv("car_data.csv")

# обучение алгоритма

# приводим строковые данные к численному формату (обучающие)
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()
city_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
drive_encoder = LabelEncoder()

data['car_brand'] = brand_encoder.fit_transform(data['car_brand'])
data['car_model'] = model_encoder.fit_transform(data['car_model'])
data['car_city'] = city_encoder.fit_transform(data['car_city'])
data['car_fuel'] = fuel_encoder.fit_transform(data['car_fuel'])
data['car_transmission'] = transmission_encoder.fit_transform(data['car_transmission'])
data['car_drive'] = drive_encoder.fit_transform(data['car_drive'])

# разделяем данные на факторы и отклики
factors = data[['car_brand', 'car_model', 'car_city', 'car_fuel', 'car_transmission', 'car_drive', 'car_mileage',
                'car_engine_capacity', 'car_engine_hp', 'car_age']]
response = data['car_price']

# производим нормализацию данных (обучающих)
scaler = MinMaxScaler()
factors = scaler.fit_transform(factors)

# разделяем на обучающую и тестовую выборки
f_train, f_test, r_train, r_test = train_test_split(factors, response, test_size=0.19999998, random_state=0)

# # регрессия случайным лесом
# regressor = RandomForestRegressor()
# regressor.fit(f_train, r_train)

# создание основного окна
window = tk.Tk()
window.title("Определитель стоимости автомобиля")

# создание текстового поля для вывода
output_text = scrolledtext.ScrolledText(window, width=80, height=30, state='disabled')
output_text.pack(pady=5)

output_text.configure(state='normal')
output_text.insert(tk.END, "Интеллектуальная система определения \nстоимости автомобиля готова к работе\nВыберите команду:\nhelp - информация по порядку и принципу ввода данных\nenter - переход к вводу данных\nexit - завершение работы системы\n")
output_text.see(tk.END)
output_text.configure(state='disabled')

def parse_drive(event):
    global cd
    cd = names.drive_type[combobox_drive.get()]
    combobox_drive.destroy()
    label_drive.destroy()

def parse_transmission(event):
    global ct
    ct = names.transmission_type[combobox_transmission.get()]
    combobox_transmission.destroy()
    label_transmission.destroy()

    global label_drive, combobox_drive
    label_drive = tk.Label(window, text="Вид привода")
    label_drive.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите вид привода\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_drive = ttk.Combobox(window, width=40, height=20, values=list(names.drive_type.keys()))
    combobox_drive.pack(pady=5)
    combobox_drive.bind("<<ComboboxSelected>>", parse_transmission)

def parse_fuel(event):
    global cf
    cf = names.engine_type[combobox_fuel.get()]
    combobox_fuel.destroy()
    label_fuel.destroy()

    global label_transmission, combobox_transmission
    label_transmission = tk.Label(window, text="Вид кпп")
    label_transmission.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите вид кпп\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_transmission = ttk.Combobox(window, width=40, height=20, values=list(names.transmission_type.keys()))
    combobox_transmission.pack(pady=5)
    combobox_transmission.bind("<<ComboboxSelected>>", parse_transmission)
def parse_city(event):
    global cci
    cci = names.cities[combobox_cities.get()]
    combobox_cities.destroy()
    label_cities.destroy()

    global label_fuel, combobox_fuel
    label_fuel = tk.Label(window, text="Тип двигателя")
    label_fuel.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите тип двигателя\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_fuel = ttk.Combobox(window, width=40, height=20, values=list(names.engine_type.keys()))
    combobox_fuel.pack(pady=5)
    combobox_fuel.bind("<<ComboboxSelected>>", parse_fuel)

def parse_model(event):
    global cm
    cm = combobox_models.get()
    combobox_models.destroy()
    label_models.destroy()

    global label_cities, combobox_cities
    label_cities = tk.Label(window, text="Город продажи")
    label_cities.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите город продажи\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_cities = ttk.Combobox(window, width=40, height=20, values=list(names.cities.keys()))
    combobox_cities.pack(pady=5)
    combobox_cities.bind("<<ComboboxSelected>>", parse_city)

def parse_brand(event):
    global cb
    cb = combobox_brands.get()
    combobox_brands.destroy()
    label_brands.destroy()

    global label_models, combobox_models
    label_models = tk.Label(window, text="Модель автомобиля")
    label_models.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите модель автомобиля\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_models = ttk.Combobox(window, width=40, height=20, values=names.models_by_brand[cb])
    combobox_models.pack(pady=5)
    combobox_models.bind("<<ComboboxSelected>>", parse_model)

def parse_command(event):
    command = combobox.get() # получение команды из выпадающего меню
    if command == "help":
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Для оценки стоимости автомобиля используются следующие параметры:\n1) Производитель: предлагается выбрать один вариант из предложенного списка, например, Volkswagen\n2) Модель: предлагается выбрать один вариант из предложенного списка, например, Polo\n3) Город продажи: предлагается выбрать один вариант из предложенного списка, например, Москва\n4) Вид топлива: предлагается выбрать один вариант из предложенного списка, например, бензин\n5) Вид трансмиссии: предлагается выбрать один вариант из предложенного списка, например, автоматическая\n6) Вид привода: предлагается выбрать один вариант из предложенного списка, например, передний\n7) Пробег: необходимо ввести целое число километров, например, 75000\n8) Рабочий объём двигателя: необходимо ввести дробное число литров, разделяя разряды точкой, например, 1.5\n9) Мощность двигателя: необходимо ввести целое число лошадиных сил, например, 110\n10) Год производства автомобиля: необоходимо ввести целоче число, например, 2018\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.pack(pady=5)
    elif command == 'exit':
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Завершение работы")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.pack(pady=5)
        window.quit()
    else:
        # сбор данных пользователя

        global label_brands, combobox_brands
        label_brands = tk.Label(window, text="Марка автомобиля")
        label_brands.pack(pady=5)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Выберите марку автомобиля\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        combobox_brands = ttk.Combobox(window, width=40, height=20, values=names.brands)
        combobox_brands.pack(pady=5)
        combobox_brands.bind("<<ComboboxSelected>>", parse_brand)


            # flag = False
            # while (flag != True):
            #     print('введите пробег: ', end='')
            #     cmi = int(input())
            #     if (cmi >= 0):
            #         flag = True
            #     else:
            #         print('данные введены некорректно, \nповторите ввод')
            # print()
            #
            # flag = False
            # while (flag != True):
            #     print('введите объем двигателя: ', end='')
            #     cec = float(input())
            #     if cec > 0 and cec < 8:
            #         flag = True
            #     else:
            #         print('данные введены некорректно, \nповторите ввод')
            # print()
            #
            # flag = False
            # while (flag != True):
            #     print('введите количество лошадиных сил: ', end='')
            #     chp = int(input())
            #     if chp > 0 and chp < 2100:
            #         flag = True
            #     else:
            #         print('данные введены некорректно, \nповторите ввод')
            # print()
            #
            # flag = False
            # while (flag != True):
            #     print('введите год производства автомобиля: ', end='')
            #     ca = 2024 - int(input())
            #     if ca >= 0 and ca < 140:
            #         flag = True
            #     else:
            #         print('данные введены некорректно, \nповторите ввод')
            # print()
            #
            # # словарь для данных пользователя
            # order_dict = {'car_brand': cb, 'car_model': cm, 'car_city': cci, 'car_fuel': cf,
            #               'car_transmission': ct, 'car_drive': cd, 'car_mileage': cmi,
            #               'car_engine_capacity': cec, 'car_engine_hp': chp, 'car_age': ca}
            #
            # # проверка введённых данных
            # print('Ваши данные:')
            # for i in order_dict.values():
            #     print(i, sep=', ', end=' ')
            # print()
            # print('Данные коректны: да/нет?')
            # key = input()
            # if (key == 'да'):
            #     flag_glob = False


label = tk.Label(window, text="Команда")
label.pack(pady=5)

# создание выпадающего меню для выбора команды
combobox = ttk.Combobox(window, width=40, height=20, values=['help', 'enter', 'exit'])
combobox.pack(pady=5)
combobox.bind("<<ComboboxSelected>>", parse_command)

window.mainloop()

# запуск системы
# while True:
#     flag_start = True
#     while flag_start:
#         print("Команда: ", end='')
#         command = input()
#         if command != "help" and command != "enter" and command != "exit":
#             print("Ввод некорректен, повторите его")
#         else:
#             flag_start = False
#     if command == "help":
#         print("Для оценки стоимости автомобиля используются следующие параметры:\n"
#               "1) Производитель: предлагается выбрать один вариант из предложенного списка, например, Volkswagen\n"
#               "2) Модель: предлагается выбрать один вариант из предложенного списка, например, Polo\n"
#               "3) Город продажи: предлагается выбрать один вариант из предложенного списка, например, Москва\n"
#               "4) Вид топлива: предлагается выбрать один вариант из предложенного списка, например, бензин\n"
#               "5) Вид трансмиссии: предлагается выбрать один вариант из предложенного списка, например, автоматическая\n"
#               "6) Вид привода: предлагается выбрать один вариант из предложенного списка, например, передний\n"
#               "7) Пробег: необходимо ввести целое число километров, например, 75000\n"
#               "8) Рабочий объём двигателя: необходимо ввести дробное число литров, разделяя разряды точкой, например, 1.5\n"
#               "9) Мощность двигателя: необходимо ввести целое число лошадиных сил, например, 110\n"
#               "10) Год производства автомобиля: необоходимо ввести целоче число, например, 2018\n")
#     elif command == "enter":
#         flag_glob = True
#         while flag_glob:
#             # сбор данных пользователя
#             flag = False
#             out(names.brands)
#             while (flag != True):
#                 print('выберите марку автомобиля: ', end='')
#                 cb = input()
#                 if cb in names.brands:
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             out(names.models_by_brand[cb])
#             while (flag != True):
#                 print('выберите модель автомобиля: ', end='')
#                 cm = input()
#                 if cm in names.models_by_brand[cb]:
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             out(names.cities.keys())
#             while (flag != True):
#                 print('выберите город продажи: ', end='')
#                 s = input()
#                 if s in names.cities.keys():
#                     cci = names.cities[s]
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             out(names.engine_type.keys())
#             while (flag != True):
#                 print('выберите вид топлива: ', end='')
#                 s = input()
#                 if s in names.engine_type.keys():
#                     cf = names.engine_type[s]
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             out(names.transmission_type.keys())
#             while (flag != True):
#                 print('выберите вид кпп: ', end='')
#                 s = input()
#                 if s in names.transmission_type.keys():
#                     ct = names.transmission_type[s]
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             out(names.drive_type.keys())
#             while (flag != True):
#                 print('выберите вид привода: ', end='')
#                 s = input()
#                 if s in names.drive_type.keys():
#                     cd = names.drive_type[s]
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             while (flag != True):
#                 print('введите пробег: ', end='')
#                 cmi = int(input())
#                 if (cmi >= 0):
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             while (flag != True):
#                 print('введите объем двигателя: ', end='')
#                 cec = float(input())
#                 if cec > 0 and cec < 8:
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             while (flag != True):
#                 print('введите количество лошадиных сил: ', end='')
#                 chp = int(input())
#                 if chp > 0 and chp < 2100:
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             flag = False
#             while (flag != True):
#                 print('введите год производства автомобиля: ', end='')
#                 ca = 2024 - int(input())
#                 if ca >= 0 and ca < 140:
#                     flag = True
#                 else:
#                     print('данные введены некорректно, \nповторите ввод')
#             print()
#
#             # словарь для данных пользователя
#             order_dict = {'car_brand': cb, 'car_model': cm, 'car_city': cci, 'car_fuel': cf,
#                           'car_transmission': ct, 'car_drive': cd, 'car_mileage': cmi,
#                           'car_engine_capacity': cec, 'car_engine_hp': chp, 'car_age': ca}
#
#             # проверка введённых данных
#             print('Ваши данные:')
#             for i in order_dict.values():
#                 print(i, sep=', ', end=' ')
#             print()
#             print('Данные коректны: да/нет?')
#             key = input()
#             if (key == 'да'):
#                 flag_glob = False
#
#         # переводим данные в dataframe
#         order_data = pd.DataFrame(order_dict, index=[0])
#
#         # приводим строковые данные к численному формату (пользовательские)
#         order_data['car_brand'] = brand_encoder.transform(order_data['car_brand'])
#         order_data['car_model'] = model_encoder.transform(order_data['car_model'])
#         order_data['car_city'] = city_encoder.transform(order_data['car_city'])
#         order_data['car_fuel'] = fuel_encoder.transform(order_data['car_fuel'])
#         order_data['car_transmission'] = transmission_encoder.transform(order_data['car_transmission'])
#         order_data['car_drive'] = drive_encoder.transform(order_data['car_drive'])
#
#         # проводим нормализацию данных (пользовательских)
#         order_data = scaler.transform(order_data)
#
#         # делаем прогноз
#         order_prediction = regressor.predict(order_data)
#         print('Оценочная стоимость автомобиля с введёнными параметрами: ', end='')
#         print(int(round(*order_prediction * 0.78, -2)), '-', int(round(*order_prediction * 0.95, -2)), 'руб.')
#     else:
#         print("Завершение работы")
#         break