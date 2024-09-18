#словарь с доступными городами
cities = {'Астрахань' : 'Astrakhan', 'Екатеринбург' : 'Yekaterinburg', 'Иркутск' : 'Irkutsk', 'Красноярск' : 'Krasnoyarsk',
          'Ростов-на-Дону' : 'Rostov-on-Don','Самара' : 'Samara', 'Санкт-Петербург' : 'Saint Petersburg',
          'Томск' : 'Tomsk', 'Челябинск' : 'Chelyabinsk', 'Чита' : 'Chita', 'Москва' : 'Moscow',
      'Новосибирск' : 'Novosibirsk', 'Якутск' : 'Yakutsk', 'Барнуаул' : 'Barnaul', 'Нижний Новгород' : 'Nizhny Novgorod',
          'Сургут' : 'Surgut', 'Уфа' : 'Ufa', 'Махачкала' : 'Makhachkala', 'Хабаровск' : 'Khabarovsk', 'Воронеж' : 'Voronezh',
          'Благовещенск' : 'Blagoveshchensk', 'Омск' : 'Omsk', 'Казань' : 'Kazan', 'Краснодар' : 'Krasnodar'}

#кортеж с доступными брендами
brands = ('Ford', 'Mercedes-Benz', 'Haval', 'Toyota', 'OMODA', 'Honda',
      'Kia', 'Hyundai', 'Renault', 'Opel', 'Great', 'Infiniti', 'Audi',
      'Volkswagen', 'GAZ', 'Peugeot', 'Mitsubishi', 'Nissan', 'Jeep',
      'Lexus', 'Mazda', 'Skoda', 'Geely', 'Chevrolet', 'Chery', 'BMW',
      'Citroen', 'Suzuki', 'Lifan', 'Porsche', 'Subaru', 'GAC', 'Cadillac',
      'Jaguar', 'Ravon', 'Jetour', 'Dodge', 'Hummer', 'Li', 'SsangYong', 'Daihatsu',
      'Chrysler', 'Land Rover', 'Rover', 'Daewoo', 'Datsun', 'EXEED', 'Changan',
      'ZAZ', 'Volvo', 'BYD', 'Vortex', 'SEAT', 'ZIL', 'IZH', 'Isuzu', 'Lada',
      'Bentley', 'MINI', 'Dongfeng', 'Acura', 'Lincoln', 'AITO', 'SWM', 'GMC',
      'RAM', 'Genesis', 'Voyah', 'Tank', 'Foton', 'Zotye', 'Lynk', 'FAW',
      'Alfa Romeo', 'Fiat', 'Lancia', 'Saab', 'Kaiyi', 'Hafei', 'Smart',
      'Tianye', 'Derways', 'JAC', 'Luxgen', 'Scion', 'BAIC', 'Rolls-Royce',
      'Ferrari', 'Maybach', 'Lamborghini', 'Roewe', 'Haima', 'Buick', 'Iran',
      'Skywell', 'Mercury', 'Livan', 'Brilliance', 'Oshan', 'Hawtai', 'Jaecoo',
      'Pontiac', 'Hongqi', 'DW', 'Maserati', 'BAW', 'Changhe', 'Moskvich',
      'TagAZ', 'UAZ', 'Tianma')

#словарь с доступными моделями конкретного бренда
models_by_brand = {'Ford' : ('B-MAX', 'Bronco', 'Bronco Sport', 'C-MAX', 'Cougar', 'EcoSport', 'Edge'
                             'Escape', 'Escort', 'Expedition', 'Explorer', 'F150', 'F250',
                             'Festiva', 'Fiesta', 'Focus', 'Focus RS', 'Fusion', 'Galaxy',
                             'Granada', 'Kuga', 'Laser', 'Maverik', 'Mondeo', 'Mustang'
                             'Orion', 'Probe', 'Ranger', 'Scorpio', 'Sierra', 'S-MAX',
                             'Taurus', 'Telstar', 'Tourneo', 'Tourneo Connect', 'Tourneo Custom'),
                    'Mercedes-Benz' : ('190', 'A-Class', 'AMG GT', 'B-Class', 'C-class', 'Citan',
                                       'CLA-Class', 'CL-Class', 'CLE-Class', 'CLK-Class', 'CLS-Class',
                                       'E-Class', 'G-Class', 'GLA-Class', 'GLB-Class', 'GLC',
                                       'GLC Coupe', 'GL-Class', 'GLE', 'GLE Coupe', 'GLK-Class',
                                       'GLS-Class', 'M-Class', 'Mercedes', 'R-Class', 'S-Class',
                                       'SLC-Class', 'SL-Class', 'SLK-Class', 'V-Class', 'Viano', 'Vito', 'X-Class'),
                    'Haval' : ('Big Dog DaGou', 'Dargo', 'F7', 'F7x', 'H2', 'H5', 'H6',
                               'H6 Coupe', 'H6S', 'H9', 'Jolion', 'M6', 'M6 Plus', 'Xialong Max'),
                    'Toyota' : ('4Runner', 'Allex', 'Allion', 'Alphard', 'Altezza', 'Aqua',
                                'Aristo', 'Auris', 'Avalon', 'Avanza', 'Avensis', 'bB', 'Belta', 'Blade',
                                'Brevis', 'Caldina', 'Cami', 'Camry', 'Camry Gracia', 'Camry Prominent',
                                'Carina', 'Carina E', 'Carina ED', 'Carina II', 'Celica', 'Celsior',
                                'Century', 'Chaser', 'C-HR', 'Corolla', 'Corolla Altis', 'Corolla Axio',
                                'Corolla Ceres', 'Corolla Cross', 'Corolla Fielder', 'Corolla FX',
                                'Corolla II', 'Corolla Levin', 'Corolla Rumion', 'Corolla Runx', 'Corolla Spacio',
                                'Corolla Verso', 'Corona', 'Corona Exiv', 'Corona Premio', 'Corona SF', 'Corsa',
                                'Cresta', 'Crown', 'Crown Majesta', 'Curren', 'Cynos', 'Duet', 'Echo', 'Esquire',
                                'Estima', 'Estima Emina', 'Estima Lucida', 'FJ Cruiser', 'Fortuner', 'Funcargo',
                                'Gaia', 'Grand Hiace', 'Granvia', 'GT 86', 'Harrier', 'Hiace', 'Hiace Regius',
                                'Highlander', 'Hilux', 'Hilux Surf', 'Ipsum', 'iQ', 'Isis', 'ist', 'Kluger V',
                                'Land Cruiser', 'Land Cruiser Cygnus', 'Land Cruiser Prado', 'Lite Ace', 'Lite Ace Noah',
                                'Mark II', 'Mark II Wagon Bilt', 'Mark II Wagon Qualis', 'Mark X', 'Mark X Zio',
                                'Master Ace Surf', 'Matrix', 'MR-S', 'Nadia', 'Noah', 'Opa', 'Passo', 'Passo Sette',
                                'Picnic', 'Pixis Epoch', 'Pixis Joy', 'Pixis Van', 'Platz', 'Porte', 'Premio',
                                'Prius', 'Prius Alpha', 'Prius PHV', 'Prius Prime', 'Probox', 'Progres', 'Ractis',
                                'Raize', 'Raum0', 'RAV4', 'Regius', 'Regius Ace', 'Roomy', 'Rush', 'Sai',
                                'Scepter', 'Sequoia', 'Sienna', 'Soarer', 'Spade', 'Sparky', 'Sprinter', 'Sprinter Carib',
                                'Sprinter Marino', 'Sprinter Trueno', 'Starlet', 'Succeed', 'Supra', 'Tacoma', 'Tank',
                                'Tercel', 'Touring Hiace', 'Town Ace', 'Town Ace Noah', 'Tundra', 'Urban Cruiser',
                                'Vanguard', 'Vellfire', 'Veloz', 'Venza', 'Verossa', 'Verso', 'Vios', 'Vista',
                                'Vista Ardeo', 'Vitz', 'Voltz', 'Voxy', 'WiLL Cypha', 'WiLL VS', 'Windom', 'Wish', 'Yaris', 'Yaris Cross'),
                    'OMODA' : ('C5', 'S5'),
                   'Honda' : ('Accord', 'Accord Inspire', 'Acty', 'Airwave', 'Ascot', 'Ascot Innova', 'Avancier',
                              'Capa', 'City', 'Civic', 'Civic Ferio', 'Crossroad', 'Crosstour', 'CR-V', 'CR-X del Sol',
                              'CR-Z', 'Domani', 'Edix', 'Element', 'Elysion', 'Fit', 'Fit Aria', 'Fit Shuttle',
                              'Freed', 'Freed Spike', 'Freed+', 'Grace', 'HR-V', 'Insight', 'Inspire', 'Integra',
                              'Integra SJ', 'Jade', 'Jazz', 'Legend', 'Life', 'Logo', 'Mobilio Spike', 'N-BOX',
                              'N-BOX Slash', 'N-ONE', 'N-VAN', 'N-WGN', 'Odyssey', 'Orthia', 'Partner', 'Pilot',
                              'Prelude', 'Rafaga', 'S660', 'Saber', 'Shuttle', 'S-MX', 'Stepwgn', 'Stream', 'Torneo',
                              'Vamos Hobio', 'Vezel', 'Vigor', 'Z', 'Zest', 'ZR-V'),
                   'Kia' : ('Avella', 'Besta', 'Carens', 'Carnival', 'Ceed', 'Cerato' , 'Cerato Koup', 'Clarus',
                            'Forte', 'K3', 'K5', 'K7', 'K8', 'K9', 'KX3', 'Magentis', 'Mohave', 'Morning', 'Niro',
                            'Opirus', 'Optima', 'Pegas', 'Picanto', 'Pregio', 'Pride', 'ProCeed', 'Quoris', 'Ray',
                            'Rio', 'Rio X (X-Line)', 'Seltos', 'Shuma', 'Sonet', 'Sorento', 'Soul', 'Spectra',
                            'Sportage', 'Stinger', 'Stonic', 'Telluride', 'Venga', 'Xceed'),
                    'Hyundai' : ('Accent', 'Avante', 'Coupe', 'Creta', 'Custo', 'Elantra', 'Equus', 'Galloper',
                                 'Genesis', 'Getz', 'Grace', 'Grand Santa Fe', 'Grand Starex', 'Grandeur', 'H1',
                                 'H200', 'i20', 'i30', 'i40', 'ix35', 'ix55', 'Kona', 'lantra', 'Lavita', 'Matrix',
                                 'Mufasa', 'NF', 'Palisade', 'Santa Fe', 'Santa Fe Classic', 'Solaris', 'Sonata',
                                 'Starex', 'Staria', 'Terracan', 'Trajet', 'Tucson', 'Veloster', 'Verna'),
                    'Renault' : ('19', 'Arkana', 'Captur', 'Clio', 'Duster', 'Espace', 'Fluence',
                                 'Grand Scenic', 'Kadjar', 'Kangoo', 'Kaptur', 'Koleos', 'Laguna', 'latitude',
                                 'Lodgy', 'Logan', 'Logan Stepway', 'Megane', 'Modus', 'Safrane', 'Samsung QM6',
                                 'Samsung SM6', 'Sandero', 'Sandero Stepway', 'Scenic', 'Symbol', 'Talisman', 'Trafic'),
                    'Opel' : ('Agila', 'Antara', 'Astra', 'Astra Family', 'Astra GTC', 'Calibra', 'Combo',
                              'Corsa', 'Crossland', 'Frontera', 'Crandland X', 'Insignia', 'Kapitan', 'Meriva', 'Mokka',
                              'Monterey', 'Omega', 'Tigra', 'Vectra', 'Vita', 'Zarifa', 'Zarifa Life'),
                    'Great' : ('Wall Deer', 'Wall Hover H3', 'Wall Hover H5', 'Wall Hover M2', 'Wall Hover M4',
                               'Wall Pao', 'Wall Poer', 'Wall Poer KingKong', 'Wall Safe', 'Wall Sing', 'Wall Wingle', 'Wall Wingle 7'),
                    'Infiniti' : ('EX25', 'EX35', 'EX37', 'FX30d', 'FX35', 'FX37', 'FX45', 'FX50', 'G25', 'G35', 'G37',
                                  'JX35', 'M35', 'M37', 'Q30', 'Q50', 'Q60','Q70', 'QX50', 'QX55', 'QX56', 'QX60', 'QX70', 'QX80'),
                    'Audi' : ('80', '100', 'A1', 'A3', 'A4', 'A4 allroad quattro', 'A5', 'A6', 'A6 allroad quattro',
                              'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q5 Sportback', 'Q7', 'Q8', 'RS Q8', 'RS5', 'RS6', 'RS7',
                              'S5', 'S8', 'SQ7', 'SQ8', 'TT', 'V8'),
                    'Volkswagen' : ('Amarok', 'Arteon', 'Atlas', 'Beetle', 'Bora', 'Caddy', 'Caravelle', 'Eos', 'Golf',
                                    'Golf Plus', 'Golf Sportsvan', 'Jetta', 'Multivan', 'Passat', 'Passat CC', 'Pointer',
                                    'Polo', 'Scirocco', 'Sharan', 'Taos', 'Tavendor', 'T-Cross', 'Teramont', 'Tiguan',
                                    'Tiguan Allspace', 'Touareg', 'Touran', 'Transporter', 'T-Roc', 'up!', 'Vento'),
                    'GAZ' : ('67', '69', '2217', '21 Volga', '24 Volga', '3102 Volga', '31029 Volga', '3110 Volga',
                             '31105 Volga', '3111 Volga', 'Pobeda', 'Volga Siber'),
                    'Peugeot' : ('107', '206', '207', '208', '301', '307', '308', '406', '407', '408', '508', '607',
                                 '2008', '3008', '4007', '4008', '5008', 'Expert', 'Partner', 'Partner Origin',
                                 'Partner Tepee', 'RCZ', 'Rifter', 'Traveller'),
                    'Mitsubishi' : ('Airtrek', 'ASX', 'Carisma', 'Challenger', 'Chariot', 'Chariot Grandis', 'Colt',
                                    'Colt Plus', 'Debonair', 'Delica', 'Delica D:2', 'Delica D:3', 'Delica D:5',
                                    'Diamante', 'Dion', 'Eclipse', 'Eclipse Cross', 'ek Custom', 'eK Sport', 'eK Wagon',
                                    'Emeraude', 'Endeavor', 'FTO', 'Galant', 'Galant Fortis', 'Grandis', 'L200', 'L300',
                                    'Lancer', 'Lancer Cedia', 'Lancer Cedia', 'Lancer Evolution', 'Legnum', 'Libero',
                                    'Minicab', 'Mirage', 'Mirage Dingo', 'Montero', 'Montero Sport', 'Outlander',
                                    'Outlander Sport', 'Pajero', 'Pajero iO', 'Pajero Junior', 'Pajero Mini',
                                    'Pajero Pinin', 'Pajero Sport', 'RVR', 'Space Gear', 'Space Runner', 'Space Star',
                                    'Space Wagon', 'Toppo', 'Xpander'),
                    'Nissan' : ('Avenir', 'Avenir Salut', 'Bassara', 'Bluebird', 'Bluebird Sylphy', 'Caravan', 'Caravan Elgrand',
                                'Cedric', 'Cefiro', 'Cima', 'Clipper', 'Cube', 'Cube Cubic', 'Datsun', 'DAYZ', 'DAYZ Roox',
                                'Dualis', 'Elgrand', 'Expert', 'Fairlady Z', 'Fuga', 'Gloria', 'GT-R', 'Homy Elgrand',
                                'Juke', 'Kicks', 'Kix', 'Lafesta', 'Largo', 'Latio', 'Laurel', 'Leopard', 'Liberta Villa',
                                'Liberty', 'March', 'Maxima', 'Micra', 'Murano', 'Navara', 'Note', 'NP300', 'NV200',
                                'NV350 Caravan', 'Otti', 'Pathfinder', 'Patrol', 'Prairie', 'Prairie Joy', 'Presage', 'Presea',
                                'President', 'Primastar', 'Primera', 'Primera Camino', 'Pulsar', 'Qashqai', 'Qashqai+2',
                                'Quest', 'Rasheen', "R'nessa", 'Rogue', 'Rogue Sport', 'Roox', 'Safari', 'Sentra',
                                'Serena', 'Silvia', 'Skyline', 'Stagea', 'Sunny', 'Sunny California', 'Sylphy', 'Teana',
                                'Terrano', 'Terrano II', 'Terrano Regulus', 'Tiida', 'Tiida Latio', 'Tino', 'Titan',
                                'Urvan', 'Vanette', 'Wingroad', 'Xterra', 'X-Trail'),
                    'Jeep' : ('Cherokee', 'Compass', 'Gladiator', 'Crand Cherokee', 'Liberty', 'Patriot',
                              'Renegade', 'Wagoneer', 'Wrangler'),
                    'Lexus' : ('CT200h', 'ES200', 'ES250', 'ES300', 'ES300h', 'ES350', 'GS250', 'GS300', 'GS300h', 'GS350',
                               'GS430', 'GS540h', 'GX460', 'GX470', 'HS250h', 'IS F', 'IS200', 'IS200t', 'IS220d', 'IS250',
                               'IS300', 'IS300h', 'IS350', 'LM350h', 'LS400', 'LS430', 'LS460', 'LS460L', 'LS500',
                               'LS500h', 'LS600h', 'LS600hL', 'LX450', 'LX450d', 'LX470', 'LX570', 'LX600', 'NX200', 'NX200t',
                               'NX250', 'NX300h', 'NX350', 'RC200t', 'RX200t', 'RX270', 'RX300', 'RX350', 'RX350L', 'RX400h',
                               'RX450h', 'RX450hL', 'RX500h', 'SC430', 'TX350', 'UX200', 'UX250h'),
                    'Mazda' : ('323', '626', '929', 'Atenza', 'Axela', 'AZ-Wagon', 'Bianto', 'Bongo', 'Bongo Brawny',
                               'Bongo Friendee', 'BT-50', 'Capella', 'CX-3', 'CX-30', 'CX-4', 'CX-5', 'CX-50', 'CX-7', 'CX-8',
                               'CX-9', 'Demio', 'Eunos Presso', 'Fanilia', 'Familia S-Wagon', 'Flair Crossover', 'Mazda2',
                               'Mazda3', 'Mazda3 MPS', 'Mazda5', 'Mazda6', 'Mazda6 MPS', 'Millenia', 'MPV', 'MX-30', 'MX-5',
                               'Premacy', 'Proceed Levante', 'Proceed Marvie', 'Roadster', 'RX-8', 'Tribute', 'Verisa', 'Xedos 6', 'Xedos 9'),
                    'Skoda' : ('Fabia', 'Felicia', 'Kamiq', 'Karoq', 'Kodiaq', 'Octavia', 'Praktik',
                               'Rapid', 'Roomster', 'Superb', 'Yeti'),
                    'Geely' : ('Atlas', 'Atlas Pro', 'Binyue', 'Boyue', 'Boyue Cool', 'Boyue L', 'Coolray', 'Emgrand',
                               'Emgrand EC7', 'Emgrand X7', 'GC6', 'GS', 'MK', 'MK Cross', 'Monjaro', 'Preface', 'Tugella FY11',
                               'Vision FC', 'Vision X6 Pro', 'Xingyue L'),
                    'Chevrolet' : ('Aveo', 'Blazer', 'Camaro', 'Captiva', 'Cavalier', 'Cobalt', 'Corvette', 'Cruze', 'Epica',
                                   'Equinox', 'Lacetti', 'Lanos', 'Malibu', 'Monza', 'MW', 'Nexia', 'Niva', 'Orlando',
                                   'Rezzo', 'Silverado', 'Spark', 'Suburban', 'Tahoe', 'Tracker', 'TrailBlazer', 'Traverse',
                                   'Trax', 'Viva', 'Volt'),
                    'Chery' : ('Amulet A15', 'Arrizo 7', 'Arrizo 8', 'Bonus 3 - A19', 'Bonus A13', 'CrossEastar', 'Explore 06', 'Fora A21',
                               'indiS S18D', 'Kimo A1', 'M11', 'Oriental Son', 'QQ Sweet', 'QQ6 S21', 'Tiggo 2', 'Tiggo 3',
                               'Tiggo 4', 'Tiggo4 Pro', 'Tiggo 5', 'Tiggo 5x', 'Tiggo 7', 'Tiggo 7 Pro', 'Tiggo 7 Pro Max',
                               'Tiggo 8', 'Tiggo 8 Pro', 'Tiggo 8 Pro e+ Hybrid', 'Tiggo 8 Pro Max', 'Tiggo 9', 'Tiggo T11', 'Very A13'),
                    'BMW' : ('340', '1-Series', '2-Series', '2-Series Active Tourer', '2-Series Gran Tourer', '3-Series', '3-Series Gran Turismo',
                             '4-Series', '5-Series', '5-Series Gran Turismo', '6-Series', '6-Series Gran Turismo', '7-Series', '8-Series',
                             'M3', 'M5', 'M8', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'XM', 'Z8'),
                    'Citroen' : ('Berlingo', 'C1', 'C2', 'C3', 'C3 Aircross', 'C3 Picasso', 'C4', 'C4 Aircross', 'C4 Picasso',
                                 'C5', 'C5 Aircross', 'C-Crosser', 'C-Elysee', 'DS3', 'DS4', 'DS5', 'DS7',
                                 'Grand C4 Picasso', 'Jumpy', 'Spacetourer'),
                    'Suzuki' : ('Aerio', 'Alto', 'Alto Lapin', 'Baleno', 'Cervo', 'Ciaz', 'Cultus', 'Cultus Crescent', 'Ertiga', 'Escudo',
                                'Every', 'Grand Escudo', 'Grand Vitara', 'Grand Vitara XL-7', 'Hustler', 'Ignis', 'Jimny', 'Jimny Sierra',
                                'Jimny Wide', 'Kei', 'Kizashi', 'Landy', 'Liana' , 'MR Wagon', 'Palette', 'Solio', 'Spacia', 'Splash',
                                'Swift', 'SX4', 'Vitara', 'Wagon R', 'Wagon R Plus', 'Xbee', 'XL7'),
                    'Lifan' : ('Breez', 'Cebrium', 'Celliya', 'Murman', 'Myway', 'Smily', 'Solani', 'X50', 'X60', 'X70'),
                    'Porsche' : ('911', 'Boxster', 'Cayenne', 'Cayenne Coupe', 'Cayman', 'Macan', 'Panamera'),
                    'Subaru' : ('B9 Tribeca', 'BRZ', 'Crosstek', 'Dex', 'Exiga', 'Forester', 'Impreza', 'Impreza WRX', 'Impreza WRX STI',
                                'Impreza XV', 'Justy', 'Legacy', 'Legacy B4', 'Legacy Lancaster', 'leone', 'Levorg', 'Outback',
                                'Pleo', 'R2', 'Stella', 'SVX', 'Traviq', 'Trezio', 'Tribeca', 'XV'),
                    'GAC' : ('GS5', 'GS8', 'Trumpchi GM8', 'Trumpchi M6', 'Trumpchi M6 Pro', 'Trumpchi M8'),
                    'Cadillac' : ('BLS', 'CT6', 'CTS', 'DeVille', 'Escalade', 'Fleetwood', 'Seville', 'SRX', 'XT4', 'XT5'),
                    'Jaguar' : ('E-Pace', 'F-Pace', 'F-Type', 'S-type', 'XE', 'XF', 'XJ', 'XK', 'X-Type'),
                    'Ravon' : ('Nexia R3', 'R2', 'R4'),
                    'Jetour' : ('Dashing', 'Traveller', 'X70', 'X70 Coupe', 'X70 Plus', 'X90 Plus'),
                    'Dodge' : ('Caliber', 'Caravan', 'Challenger', 'Charger', 'Durango', 'Grand Caravan', 'Intrepid', 'Journey', 'Magnum', 'Ram', 'Stealth', 'Stratus'),
                    'Hummer' : ('H2', 'H3'),
                    'Li' : ('L7', 'L8', 'L9', 'ONE'),
                    'SsangYong' : ('Actyon', 'Actyon Sports', 'Istana', 'Korando', 'Korando Family', 'Korando Sports', 'Korando Turismo',
                                   'Kyron', 'Musso Sports', 'Rexton', 'Rexton Sports', 'Rexton Sports Khan', 'Stavic', 'Torres'),
                    'Daihatsu' : ('Atrai', 'Atrai7', 'Be-Go', 'Boon', 'Cast', 'Coo', 'Copen', 'Esse', 'Hijet', 'Max', 'Mebius',
                                  'Mira', 'Mira e:S', 'Mira Gino', 'Move', 'Move Canbus', 'Pyzar', 'Rocky', 'Sonica', 'Taft', 'Tanto',
                                  'Terios', 'Terios Kid', 'Thor', 'Wake', 'YRV'),
                    'Chrysler' : ('300C', 'Concorde', 'Crossfire', 'Grand Voyager', 'Pacifica', 'PT Cruiser', 'Sebring',
                                  'Town and Country', 'Voyager'),
                    'Land Rover' : ('Rover Defender', 'Rover Discovery', 'Rover Discovery Sport', 'Rover Freelander', 'Rover Range Rover',
                                    'Rover Range Rover Evoque', 'Rover Range Rover Sport', 'Rover Range Rover Velar'),
                    'Rover' : ('45', '75', '200', '400', '600'),
                    'Daewoo' : ('Damas', 'Espero', 'Gentra', 'Lacetti', 'Lanos', 'Matiz', 'Nexia', 'Tacuma', 'Winstorm'),
                    'Datsun' : ('mi-Do', 'on-DO'),
                    'EXEED' : ('LX', 'RX', 'TXL', 'VX'),
                    'Changan' : ('Alsvin', 'CS35', 'CS35 Plus', 'CS55', 'CS55 Plus', 'CS75', 'CS75 Plus',
                                 'CS85 Coupe', 'CS95', 'CS95 Plus', 'Deepal Shenlan S7', 'Eado', 'Hunter', 'Hunter Plus',
                                 'UNI-K', 'UNI-T', 'UNI-V'),
                    'ZAZ' : ('Cance', 'Sens', 'Slavuta', 'Tavria', 'Vida', 'Zaporozhets'),
                    'Volvo' : ('240', '740', '760', '940', '960', 'C30', 'S40', 'S60', 'S80',
                               'S90', 'V40', 'V50', 'V60', 'V90', 'XC40', 'XC60', 'XC70', 'XC90'),
                    'BYD' : ('F3', 'Flyer', 'Song Plus', 'Tang', 'Yangwang U8'),
                    'Vortex' : ('Corda', 'Estina', 'Tingo'),
                    'SEAT' : ('Altea', 'Ateca', 'Ibiza', 'Leon', 'Toledo'),
                    'ZIL' : ('4104'),
                    'IZH' : ('2715', '2717', '2125 Kombi', '2126 Oda'),
                    'Isuzu' : ('Aska', 'Axiom', 'Bighorn', 'D-MAX', 'Gemini', 'MU', 'Rodeo', 'Trooper'),
                    'Lada' : ('2101', '2102', '2103', '2104', '2105', '2106', '2107', '2108', '2109', '2110', '2111', '2112',
                              '21099', '1111 Oka', '2113 Samara', '2114 Samara', '2115 Samara', '4x4 2121 Niva',
                              '4x4 2131 Niva', '4x4 Bronto', 'Granta', 'Granta Cross', 'Granta Sport', 'Kalina', 'Kalina Cross',
                              'Kalina Sport', 'Largus', 'Largus Cross', 'Niva', 'Niva Bronto', 'Niva Legend', 'Niva Travel', 'Priora',
                              'Vesta', 'Vesta Cross', 'Vesta Sport', 'XRay', 'XRay Cross'),
                    'Bentley' : ('Bentayga', 'Continental', 'Continental GT', 'Flying Spur', 'Mulsanne'),
                    'MINI' : ('Clubman', 'Countryman', 'Hatch', 'Paceman'),
                    'Dongfeng' : ('S80', 'Fengon 500', 'H30 Cross', 'S30', 'Shine Max'),
                    'Acura' : ('MDX', 'RDX', 'TLX', 'TSX', 'ZDX'),
                    'Lincoln' : ('Aviator', 'Corsair', 'MKX', 'Navigator', 'Town Car'),
                    'AITO' : ('M5', 'M7'),
                    'SWM' : ('Tiger'),
                    'GMC' : ('Canyon', 'Envoy', 'Sierra', 'Yukon'),
                    'RAM' : ('1500'),
                    'Genesis' : ('G70', 'G80', 'G90', 'GV80'),
                    'Voyah' : ('Free', 'Dream'),
                    'Tank' : ('300', '400', '500'),
                    'Foton' : ('Sauvana', 'Tunland', 'Tunland G7'),
                    'Zotye' : ('Coupa', 'T600'),
                    'Lynk' : ('& Co 01', '& Co 08', '& Co 09'),
                    'FAW' : ('Bestune T77', 'Bestune T99', 'Besturn X40', 'Besturn X80', 'V5'),
                    'Alfa Romeo' : ('Romeo 156', 'Romeo Giulia'),
                    'Fiat' : ('500', 'Albea', 'Brava', 'Bravo', 'Doblo', 'Fullback', 'Grande Punto', 'Linea',
                              'Marea', 'Palio', 'Panda', 'Punto', 'Qubo', 'Stilo', 'Tipo'),
                    'Lancia' : ('Delta'),
                    'Saab' : ('900', '9000', '9-3'),
                    'Kaiyi' : ('X3', 'E5'),
                    'Hafei' : ('Brio'),
                    'Smart' : ('Fortwo', 'City'),
                    'Tianye' : ('Admiral'),
                    'Derways' : ('Shuttle'),
                    'JAC' : ('J5', 'J7', 'JS4', 'JS6', 'S3', 'S5', 'T6'),
                    'Luxgen' : ('7 SUV'),
                    'Scion' : ('tC', 'xA'),
                    'BAIC' : ('X7', 'X35', 'U5 Plus'),
                    'Rolls-Royce' : ('Phantom'),
                    'Ferrari' : ('F8 Tributo', 'SF90'),
                    'Maybach' : ('57'),
                    'Lamborghini' : ('Aventador', 'Urus'),
                    'Roewe' : ('RX5'),
                    'Haima' : ('3'),
                    'Buick' : ('Regal'),
                    'Iran' : ('Khodro Samand'),
                    'Skywell' : ('HT-i'),
                    'Mercury' : ('Grand Marquis', 'Mariner'),
                    'Livan' : ('X3 Pro'),
                    'Brilliance' : ('H530', 'V5'),
                    'Oshan' : ('X5', 'X5 Plus', 'Z6'),
                    'Hawtai' : ('Boliger'),
                    'Jaecoo' : ('J7'),
                    'Pontiac' : ('Sunfire', 'Vibe', 'Solstice'),
                    'Hongqi' : ('HS5'),
                    'DW' : ('Hower H3'),
                    'Maserati' : ('Quattroporte'),
                    'BAW' : ('Ace M7'),
                    'Changhe' : ('Ideal'),
                    'Moskvich' : ('3', '407', '408', '412', '2140', '2141'),
                    'TagAZ' : ('Road Partner', 'Tager', 'Vega'),
                    'UAZ' : ('469', '3151', '3153', 'Buhanka', 'Hunter', 'Patriot'),
                    'Tianma' : ('Century')
}

#словарь с видами двигателей
engine_type = {'бензин' : 'gasoline', 'дизель' : 'diesel', 'гибрид' : 'hybrid'}

#словарь с видами трансмисси
transmission_type = {'механическая' : 'manual', 'автоматическая' : 'automatic', 'роботизированная' : 'robot', 'вариаторная' : 'CVT'}

#словарь с видами привода
drive_type = {'передний' : 'FWD', 'задний' : 'RWD', 'полный' : "4WD"}