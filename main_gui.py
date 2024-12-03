import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import names
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk


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

# регрессия случайным лесом
regressor = RandomForestRegressor()
regressor.fit(f_train, r_train)

# создание основного окна
window = tk.Tk()
window.title("Определитель стоимости автомобиля")

# создание текстового поля для вывода
output_text = scrolledtext.ScrolledText(window, width=80, height=15, state='disabled')
output_text.pack(pady=5)

output_text.configure(state='normal')
output_text.insert(tk.END, "Интеллектуальная система определения \nстоимости автомобиля готова к работе\nВыберите команду:\nhelp - информация по порядку и принципу ввода данных\nenter - переход к вводу данных\nexit - завершение работы системы\n")
output_text.see(tk.END)
output_text.configure(state='disabled')

def get_price():
    global cb, cm, cci, cf, ct, cd, cmi, cec, chp, ca
    order_dict = {'car_brand': cb, 'car_model': cm, 'car_city': cci, 'car_fuel': cf,
                               'car_transmission': ct, 'car_drive': cd, 'car_mileage': cmi,
                               'car_engine_capacity': cec, 'car_engine_hp': chp, 'car_age': ca}
    # переводим данные в dataframe
    order_data = pd.DataFrame(order_dict, index=[0])

    # приводим строковые данные к численному формату (пользовательские)
    order_data['car_brand'] = brand_encoder.transform(order_data['car_brand'])
    order_data['car_model'] = model_encoder.transform(order_data['car_model'])
    order_data['car_city'] = city_encoder.transform(order_data['car_city'])
    order_data['car_fuel'] = fuel_encoder.transform(order_data['car_fuel'])
    order_data['car_transmission'] = transmission_encoder.transform(order_data['car_transmission'])
    order_data['car_drive'] = drive_encoder.transform(order_data['car_drive'])

    # проводим нормализацию данных (пользовательских)
    order_data = scaler.transform(order_data)

    # делаем прогноз
    order_prediction = regressor.predict(order_data)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Оценочная стоимость автомобиля с введёнными параметрами: " + str(int(round(*order_prediction * 0.78, -2))) + '-' + str(int(round(*order_prediction * 0.95, -2))) + 'руб.\n')
    output_text.see(tk.END)
    output_text.configure(state='disabled')

    number_entry.delete(0, tk.END)
    number_entry.destroy()
def parse_year(event):
    global ca
    ca = 2024 - int(number_entry.get())
    if ca < 0 or ca > 140:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите год производства снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_year)
    else:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённый год производства: " + str(2024 - ca) + "\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        get_price()

def parse_horse_power_error(event):
    global chp
    chp = int(number_entry.get())
    if chp <= 0 or chp >= 2100:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите количество лошадиных сил снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_horse_power_error)
    else:
        number_entry.delete(0, tk.END)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённое количество лошадиных сил: " + str(chp) + "\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите год производства. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.bind('<Return>', parse_year)

def parse_horse_power(event):
    global chp
    chp = int(number_entry.get())
    if chp <= 0 or chp >= 2100:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите количество лошадиных сил снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_horse_power_error)
    else:
        number_entry.delete(0, tk.END)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённое количество лошадиных сил: " + str(chp) + "\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите год производства. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.bind('<Return>', parse_year)

def parse_engine_capacity_error(event):
    global cec
    cec = float(number_entry.get())
    if cec < 0 or cec > 8:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите объём двигателя снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_engine_capacity_error)
    else:
        number_entry.delete(0, tk.END)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённый объём двигателя: " + str(cec) + "л\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите количество лошадиных сил. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.bind('<Return>', parse_horse_power)
def parse_engine_capacity(event):
    global cec
    cec = float(number_entry.get())
    if cec < 0 or cec > 8:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите объём двигателя снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_engine_capacity_error)
    else:
        number_entry.delete(0, tk.END)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённый объём двигаетля: " + str(cec) + "л\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите количество лошадиных сил. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.bind('<Return>', parse_horse_power)

def parse_milage_error(event):
    global cmi
    cmi = int(number_entry.get())
    if cmi < 0:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите пробег снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_milage_error)
    else:
        number_entry.delete(0, tk.END)

        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённый пробег: " + str(cmi) + "км\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите объём двигателя. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')

        number_entry.bind('<Return>', parse_engine_capacity)

def parse_milage(event):
    global cmi
    cmi = int(number_entry.get())
    if cmi < 0:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Некорректный ввод, введите пробег снова\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        number_entry.delete(0, tk.END)
        number_entry.bind('<Return>', parse_milage_error)
    else:
        number_entry.delete(0, tk.END)
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введённый пробег: " + str(cmi) + "км\n")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Введите объём двигателя. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')

        number_entry.bind('<Return>', parse_engine_capacity)

def parse_drive(event):
    global cd
    s = combobox_drive.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранный тип привода: " + s + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    cd = names.drive_type[s]
    combobox_drive.destroy()
    label_drive.destroy()

    global number_entry
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Введите пробег. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    number_entry = tk.Entry(window)
    number_entry.pack(pady=5)
    number_entry.bind('<Return>', parse_milage)

def parse_transmission(event):
    global ct
    s = combobox_transmission.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранный тип кпп: " + s + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    ct = names.transmission_type[s]
    combobox_transmission.destroy()
    label_transmission.destroy()

    global label_drive, combobox_drive
    label_drive = tk.Label(window, text="Вид привода")
    label_drive.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите вид привода. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_drive = ttk.Combobox(window, width=40, height=20, values=sorted(list(names.drive_type.keys())))
    combobox_drive.pack(pady=5)
    combobox_drive.bind("<<ComboboxSelected>>", parse_drive)

def parse_fuel(event):
    global cf
    s = combobox_fuel.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранный тип двигателя: " + s + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    cf = names.engine_type[s]
    combobox_fuel.destroy()
    label_fuel.destroy()

    global label_transmission, combobox_transmission
    label_transmission = tk.Label(window, text="Вид кпп")
    label_transmission.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите вид кпп. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_transmission = ttk.Combobox(window, width=40, height=20, values=sorted(list(names.transmission_type.keys())))
    combobox_transmission.pack(pady=5)
    combobox_transmission.bind("<<ComboboxSelected>>", parse_transmission)
def parse_city(event):
    global cci
    s = combobox_cities.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранный город: " + s + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    cci = names.cities[s]
    combobox_cities.destroy()
    label_cities.destroy()

    global label_fuel, combobox_fuel
    label_fuel = tk.Label(window, text="Тип двигателя")
    label_fuel.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите тип двигателя. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_fuel = ttk.Combobox(window, width=40, height=20, values=sorted(list(names.engine_type.keys())))
    combobox_fuel.pack(pady=5)
    combobox_fuel.bind("<<ComboboxSelected>>", parse_fuel)

def parse_model(event):
    global cm
    cm = combobox_models.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранная модель: " + cm + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_models.destroy()
    label_models.destroy()

    global label_cities, combobox_cities
    label_cities = tk.Label(window, text="Город продажи")
    label_cities.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите город продажи. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_cities = ttk.Combobox(window, width=40, height=20, values=sorted(list(names.cities.keys())))
    combobox_cities.pack(pady=5)
    combobox_cities.bind("<<ComboboxSelected>>", parse_city)

def parse_brand(event):
    global cb
    cb = combobox_brands.get()
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выбранная марка: " + cb + "\n")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_brands.destroy()
    label_brands.destroy()

    global label_models, combobox_models
    label_models = tk.Label(window, text="Модель автомобиля")
    label_models.pack(pady=5)
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Выберите модель автомобиля. ")
    output_text.see(tk.END)
    output_text.configure(state='disabled')
    combobox_models = ttk.Combobox(window, width=40, height=20, values=sorted(names.models_by_brand[cb]))
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
        output_text.insert(tk.END, "Выберите марку автомобиля. ")
        output_text.see(tk.END)
        output_text.configure(state='disabled')
        combobox_brands = ttk.Combobox(window, width=40, height=20, values=sorted(names.brands))
        combobox_brands.pack(pady=5)
        combobox_brands.bind("<<ComboboxSelected>>", parse_brand)


label = tk.Label(window, text="Команда")
label.pack(pady=5)

# создание выпадающего меню для выбора команды
combobox = ttk.Combobox(window, width=40, height=20, values=['help', 'enter', 'exit'])
combobox.pack(pady=5)
combobox.bind("<<ComboboxSelected>>", parse_command)

window.mainloop()