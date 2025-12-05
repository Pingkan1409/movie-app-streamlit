{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "da467d0d",
   "metadata": {},
   "source": [
    "## <div align=\"center\"> TUGAS LAB IS388 Data Analysis </div>\n",
    "## <div align=\"center\"> WEEK [13]: [UNGUIDED]</div>\n",
    "#### <div align=\"center\"> Semester Ganjil 2025/2026 </div>\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1529afb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime\n",
    "import uuid\n",
    "\n",
    "studentName = \"Pingkan Nonie Umboh\"\n",
    "studentNIM = \"00000108758\"\n",
    "studentClass = \"EL\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c0fdf3fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Name: \t\tPingkan Nonie Umboh\n",
      "NIM: \t\t00000108758\n",
      "NIM: \t\tEL\n",
      "Start: \t\t2025-12-05 16:18:15.967929\n",
      "Device ID: \t54c39742-d1bb-11f0-a1ec-cbe2c546a480\n"
     ]
    }
   ],
   "source": [
    "myDate = datetime.datetime.now()\n",
    "myDevice = str(uuid.uuid1())\n",
    "\n",
    "print(\"Name: \\t\\t{}\".format(studentName))\n",
    "print(\"NIM: \\t\\t{}\".format(studentNIM))\n",
    "print(\"NIM: \\t\\t{}\".format(studentClass))\n",
    "print(\"Start: \\t\\t{}\".format(myDate))\n",
    "print(\"Device ID: \\t{}\".format(myDevice))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "247cf3e5",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67531a34",
   "metadata": {},
   "source": [
    "## <div align=\"center\"> Unguided Lab </div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b4f49c6",
   "metadata": {},
   "source": [
    "### Enter unguided Lab Code Here:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "54f27d0f-e91f-41ac-bb20-02163b85eeea",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import ast\n",
    "import re\n",
    "from pathlib import Path\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fab90923-75fb-40ed-867c-f79af399eec2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c6a6b9b7-7d07-40e0-8a1d-0a8c881ae38a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set_style(\"whitegrid\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "15b6735c-700a-4c55-b21f-48d86c7dd77a",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_PATH = \"final_dataset.csv\"   # letakkan file CSV di folder kerja\n",
    "OUT_DIR = Path(\"models_pdf_pipeline\")\n",
    "OUT_DIR.mkdir(exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e6fe3175-a30b-4b1b-a9b4-0ae4f42c5c44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 63249 entries, 0 to 63248\n",
      "Data columns (total 23 columns):\n",
      " #   Column                 Non-Null Count  Dtype  \n",
      "---  ------                 --------------  -----  \n",
      " 0   id                     63249 non-null  object \n",
      " 1   title                  63249 non-null  object \n",
      " 2   duration               61174 non-null  object \n",
      " 3   mpa                    41227 non-null  object \n",
      " 4   rating                 59181 non-null  float64\n",
      " 5   votes                  59181 non-null  object \n",
      " 6   méta_score             15533 non-null  float64\n",
      " 7   description            60889 non-null  object \n",
      " 8   movie_link             63249 non-null  object \n",
      " 9   writers                62980 non-null  object \n",
      " 10  directors              63198 non-null  object \n",
      " 11  stars                  62905 non-null  object \n",
      " 12  budget                 15359 non-null  object \n",
      " 13  opening_weekend_gross  16837 non-null  object \n",
      " 14  gross_worldwide        20722 non-null  object \n",
      " 15  gross_us_canada        19544 non-null  object \n",
      " 16  release_date           63249 non-null  object \n",
      " 17  countries_origin       63150 non-null  object \n",
      " 18  filming_locations      46395 non-null  object \n",
      " 19  production_companies   61276 non-null  object \n",
      " 20  awards_content         0 non-null      float64\n",
      " 21  genres                 62462 non-null  object \n",
      " 22  languages              62919 non-null  object \n",
      "dtypes: float64(3), object(20)\n",
      "memory usage: 11.1+ MB\n"
     ]
    }
   ],
   "source": [
    "movies = pd.read_csv(DATA_PATH)\n",
    "movies.shape, movies.columns\n",
    "# quick info (to inspect)\n",
    "movies.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "be5765d3-76ad-4354-a070-75421179d196",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "movies (63249, 23)\n"
     ]
    }
   ],
   "source": [
    "for var_name in dir():\n",
    "    try:\n",
    "        obj = eval(var_name)\n",
    "        if hasattr(obj, \"shape\") and hasattr(obj, \"columns\"):\n",
    "            print(var_name, obj.shape)\n",
    "    except:\n",
    "        pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "75113fe4-565a-4a9d-a7f3-bc491059df5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<function id(obj, /)>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "ae0c2c79-610b-4db9-9676-e20e3fbf10ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies = pd.read_csv(\"final_dataset.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2a286395-1250-46b8-93c3-5187c8a73090",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Kolom lengkap. Lanjut!\n"
     ]
    }
   ],
   "source": [
    "required_columns = [\"budget\", \"gross_worldwide\", \"rating\", \"votes\"]\n",
    "\n",
    "for c in required_columns:\n",
    "    if c not in movies.columns:\n",
    "        raise Exception(\n",
    "            f\"Required column '{c}' not found.\\nKolom yang tersedia: {movies.columns.tolist()}\"\n",
    "        )\n",
    "\n",
    "print(\"Kolom lengkap. Lanjut!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "6eef735d-bc96-4bd0-afd2-37eb7962ed8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method DataFrame.info of               id                       title duration        mpa  rating  \\\n",
       "0      tt0027483          The Crimson Circle   1h 16m        NaN     6.4   \n",
       "1      tt0058131  The Mystery of Thug Island   1h 36m        NaN     5.0   \n",
       "2      tt0042760   Las mujeres de mi general   1h 52m  Not Rated     6.8   \n",
       "3      tt0027667                Gentle Julia    1h 2m   Approved     6.8   \n",
       "4      tt0055747              Love at Twenty   1h 50m        NaN     7.2   \n",
       "...          ...                         ...      ...        ...     ...   \n",
       "63244  tt0160310           Girl with an Itch   1h 17m   Approved     6.4   \n",
       "63245  tt0103069                  Texasville    2h 6m          R     6.0   \n",
       "63246  tt1473380                 Kalai Arasi    2h 3m        NaN     8.0   \n",
       "63247  tt0247184                      Aprili      45m        NaN     7.2   \n",
       "63248  tt0069860   Cebo para una adolescente   1h 28m        NaN     4.8   \n",
       "\n",
       "      votes  méta_score                                        description  \\\n",
       "0        30         NaN  An extortion ring murders anyone who refuses t...   \n",
       "1       114         NaN  Three year old Ada, daughter of the British ca...   \n",
       "2        74         NaN  Infante stars as a rebel general caught up in ...   \n",
       "3        38         NaN  A shy newspaperman (Brown) nearly gives up whe...   \n",
       "4      2.5K         NaN  \"Love at Twenty\" unites five directors from ar...   \n",
       "...     ...         ...                                                ...   \n",
       "63244    63         NaN  A lonely, widowed middle-aged ranch owner give...   \n",
       "63245  3.2K        49.0  The summer of 1984: 32 years after Duane Jacks...   \n",
       "63246    56         NaN  Vani is kidnapped by aliens who need her to te...   \n",
       "63247   522         NaN  A young happy couple moves from a poor distric...   \n",
       "63248   156         NaN  Maribel, a typist, becomes the director's secr...   \n",
       "\n",
       "                                  movie_link  \\\n",
       "0      https://www.imdb.com/title/tt0027483/   \n",
       "1      https://www.imdb.com/title/tt0058131/   \n",
       "2      https://www.imdb.com/title/tt0042760/   \n",
       "3      https://www.imdb.com/title/tt0027667/   \n",
       "4      https://www.imdb.com/title/tt0055747/   \n",
       "...                                      ...   \n",
       "63244  https://www.imdb.com/title/tt0160310/   \n",
       "63245  https://www.imdb.com/title/tt0103069/   \n",
       "63246  https://www.imdb.com/title/tt1473380/   \n",
       "63247  https://www.imdb.com/title/tt0247184/   \n",
       "63248  https://www.imdb.com/title/tt0069860/   \n",
       "\n",
       "                                                 writers  ...  \\\n",
       "0      ['Reginald Denham', 'Edgar Wallace', 'Howard I...  ...   \n",
       "1      ['Emilio Salgari', 'Arpad DeRiso', 'Ottavio Po...  ...   \n",
       "2      ['Joselito Rodríguez', 'Celestino Gorostiza', ...  ...   \n",
       "3                   ['Booth Tarkington', 'Lamar Trotti']  ...   \n",
       "4      ['Shintarô Ishihara', 'Marcel Ophüls', 'Renzo ...  ...   \n",
       "...                                                  ...  ...   \n",
       "63244  ['Ralph S. Whiting', 'Peter Perry Jr.', 'E. Sh...  ...   \n",
       "63245            ['Larry McMurtry', 'Peter Bogdanovich']  ...   \n",
       "63246                               ['T.E. Gnanamurthy']  ...   \n",
       "63247           ['Erlom Akhvlediani', 'Otar Iosseliani']  ...   \n",
       "63248  ['Francisco Lara Polop', 'Francisco Summers', ...  ...   \n",
       "\n",
       "      opening_weekend_gross gross_worldwide gross_us_canada release_date  \\\n",
       "0                       NaN             NaN             NaN   1936-08-10   \n",
       "1                       NaN             NaN             NaN   1966-05-28   \n",
       "2                       NaN             NaN             NaN   1951-07-13   \n",
       "3                       NaN             NaN             NaN   1936-04-10   \n",
       "4                       NaN             NaN             NaN   1963-02-06   \n",
       "...                     ...             ...             ...          ...   \n",
       "63244                   NaN             NaN             NaN   1958-11-12   \n",
       "63245              $823,534      $2,268,181      $2,268,181   1990-09-28   \n",
       "63246                   NaN             NaN             NaN   1963-04-19   \n",
       "63247                   NaN             NaN             NaN   1962-09-28   \n",
       "63248                   NaN             NaN             NaN   1982-02-08   \n",
       "\n",
       "                                        countries_origin  \\\n",
       "0                                     ['United Kingdom']   \n",
       "1                    ['Italy', 'Monaco', 'West Germany']   \n",
       "2                                             ['Mexico']   \n",
       "3                                      ['United States']   \n",
       "4      ['France', 'Italy', 'Japan', 'Poland', 'West G...   \n",
       "...                                                  ...   \n",
       "63244                                  ['United States']   \n",
       "63245                                  ['United States']   \n",
       "63246                                          ['India']   \n",
       "63247                                   ['Soviet Union']   \n",
       "63248                                          ['Spain']   \n",
       "\n",
       "                                       filming_locations  \\\n",
       "0                                                    NaN   \n",
       "1                                                    NaN   \n",
       "2                                                    NaN   \n",
       "3      ['20th Century Fox Studios - 10201 Pico Blvd.,...   \n",
       "4      ['Warsaw Zoo, Ratuszowa, Praga Pólnoc, Warsaw,...   \n",
       "...                                                  ...   \n",
       "63244            ['Los Angeles County, California, USA']   \n",
       "63245                        ['Archer City, Texas, USA']   \n",
       "63246                                                NaN   \n",
       "63247                                                NaN   \n",
       "63248           ['Parque de Atracciones, Madrid, Spain']   \n",
       "\n",
       "                                    production_companies awards_content  \\\n",
       "0                     ['Richard Wainwright Productions']            NaN   \n",
       "1                        ['Eichberg-Film', 'Liber Film']            NaN   \n",
       "2                    ['Producciones Rodríguez Hermanos']            NaN   \n",
       "3                              ['Twentieth Century Fox']            NaN   \n",
       "4      ['Ulysse Productions', 'Unitec Films', 'Cinese...            NaN   \n",
       "...                                                  ...            ...   \n",
       "63244                            ['The Don-Tru Company']            NaN   \n",
       "63245            ['Cine-Source', 'Nelson Entertainment']            NaN   \n",
       "63246                                ['Sarodi Brothers']            NaN   \n",
       "63247  ['Georgian-Film', 'Gruziya Film', 'Qartuli Pil...            NaN   \n",
       "63248  ['Producciones Internacionales Cinematográfica...            NaN   \n",
       "\n",
       "                               genres  \\\n",
       "0                           ['Drama']   \n",
       "1                       ['Adventure']   \n",
       "2                    ['Drama', 'War']   \n",
       "3      ['Comedy', 'Drama', 'Romance']   \n",
       "4                ['Drama', 'Romance']   \n",
       "...                               ...   \n",
       "63244                       ['Drama']   \n",
       "63245            ['Drama', 'Romance']   \n",
       "63246           ['Comedy', 'Fantasy']   \n",
       "63247  ['Comedy', 'Drama', 'Romance']   \n",
       "63248  ['Comedy', 'Drama', 'Romance']   \n",
       "\n",
       "                                               languages  \n",
       "0                                            ['English']  \n",
       "1                                            ['Italian']  \n",
       "2                                            ['Spanish']  \n",
       "3                                            ['English']  \n",
       "4      ['French', 'Polish', 'Japanese', 'Italian', 'G...  \n",
       "...                                                  ...  \n",
       "63244                                        ['English']  \n",
       "63245                                        ['English']  \n",
       "63246                                          ['Tamil']  \n",
       "63247                                       ['Georgian']  \n",
       "63248                                        ['Spanish']  \n",
       "\n",
       "[63249 rows x 23 columns]>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies.info"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "9be86679-bb6c-4309-91cd-58086ec53ebd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(63249, 13)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "high_missing_cols = [\n",
    "    \"awards_content\",\n",
    "    \"méta_score\",\n",
    "    \"opening_weekend_gross\",\n",
    "    \"gross_us_canada\",\n",
    "    \"filming_locations\",\n",
    "    \"mpa\"\n",
    "]\n",
    "irrelevant_cols = [\n",
    "    \"countries_origin\", \"languages\", \"movie_link\", \"production_countries\", \"description\"\n",
    "]\n",
    "\n",
    "movies_clean = movies.drop(columns=high_missing_cols + irrelevant_cols, errors='ignore')\n",
    "movies_clean.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a32e13e8-997a-489f-9f53-18ebc305291d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10413, 13)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "required = [\"title\", \"budget\", \"gross_worldwide\", \"release_date\"]\n",
    "movies_clean = movies_clean.dropna(subset=required)\n",
    "movies_clean.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ed85c1c6-2d42-4e38-b901-709f6fa9b27b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: cleaning helper functions exactly like PDF semantics\n",
    "\n",
    "def clean_duration(x):\n",
    "    if pd.isna(x):\n",
    "        return None\n",
    "    x = str(x)\n",
    "    hours = 0\n",
    "    minutes = 0\n",
    "    if \"h\" in x:\n",
    "        try:\n",
    "            hours = int(x.split(\"h\")[0].strip())\n",
    "        except:\n",
    "            hours = 0\n",
    "    if \"m\" in x:\n",
    "        try:\n",
    "            minutes = int(x.split(\"h\")[-1].replace(\"m\", \"\").strip())\n",
    "        except:\n",
    "            minutes = 0\n",
    "    return hours * 60 + minutes\n",
    "\n",
    "def clean_money(x):\n",
    "    # PDF method: remove \"$\" and \",\" then keep first token and to_numeric\n",
    "    if pd.isna(x):\n",
    "        return None\n",
    "    x = str(x)\n",
    "    x = x.replace(\"$\", \"\").replace(\",\", \"\")\n",
    "    x = x.split()[0]  # keep only first token before any text\n",
    "    return pd.to_numeric(x, errors=\"coerce\")\n",
    "\n",
    "def clean_votes(x):\n",
    "    if pd.isna(x):\n",
    "        return None\n",
    "    x = str(x).lower().strip().replace(\",\", \"\")\n",
    "    if \"k\" in x:\n",
    "        try:\n",
    "            return int(float(x.replace(\"k\",\"\")) * 1000)\n",
    "        except:\n",
    "            return pd.to_numeric(x, errors=\"coerce\")\n",
    "    return pd.to_numeric(x, errors=\"coerce\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "1e347e74-0447-48c5-9186-27b860e45460",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: apply cleaning (duration, money fields, votes, release_date)\n",
    "movies_clean[\"duration\"] = movies_clean[\"duration\"].apply(clean_duration)\n",
    "movies_clean[\"budget\"] = movies_clean[\"budget\"].apply(clean_money)\n",
    "movies_clean[\"gross_worldwide\"] = movies_clean[\"gross_worldwide\"].apply(clean_money)\n",
    "# other money columns (if present)\n",
    "if \"opening_weekend_gross\" in movies_clean.columns:\n",
    "    movies_clean[\"opening_weekend_gross\"] = movies_clean[\"opening_weekend_gross\"].apply(clean_money)\n",
    "if \"gross_us_canada\" in movies_clean.columns:\n",
    "    movies_clean[\"gross_us_canada\"] = movies_clean[\"gross_us_canada\"].apply(clean_money)\n",
    "\n",
    "movies_clean[\"votes\"] = movies_clean[\"votes\"].apply(clean_votes)\n",
    "movies_clean[\"release_date\"] = pd.to_datetime(movies_clean[\"release_date\"], errors=\"coerce\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "7e6ef657-47c2-45a2-9f21-c4f4b793bc79",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 7: fill numeric missing with median (duration, budget, gross_worldwide, votes) as PDF\n",
    "for col in [\"duration\", \"budget\", \"gross_worldwide\", \"votes\"]:\n",
    "    if col in movies_clean.columns:\n",
    "        movies_clean[col] = movies_clean[col].fillna(movies_clean[col].median())\n",
    "\n",
    "# fill categorical columns with 'Unknown' as PDF\n",
    "cat_cols = [\"writers\", \"directors\", \"stars\", \"production_companies\", \"genres\"]\n",
    "for col in cat_cols:\n",
    "    if col in movies_clean.columns:\n",
    "        movies_clean[col] = movies_clean[col].fillna(\"Unknown\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "6871d9eb-e523-4565-8442-0ad876bae07d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 8: clean list-like strings: convert \"['A','B']\" -> \"A, B\" (PDF uses ast.literal_eval then join, but then they take first element)\n",
    "def clean_list_string(x):\n",
    "    if pd.isna(x):\n",
    "        return \"Unknown\"\n",
    "    try:\n",
    "        items = ast.literal_eval(x)\n",
    "        if isinstance(items, (list, tuple)) and len(items) > 0:\n",
    "            return \", \".join([str(i).strip() for i in items])\n",
    "        else:\n",
    "            return str(x)\n",
    "    except:\n",
    "        return str(x)\n",
    "\n",
    "# apply to list-like columns then in later steps we'll extract first element\n",
    "for col in [\"writers\", \"directors\", \"stars\", \"production_companies\", \"genres\"]:\n",
    "    if col in movies_clean.columns:\n",
    "        movies_clean[col] = movies_clean[col].apply(clean_list_string)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3b009998-4c48-4284-a460-161d442bc411",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 9: Extract first listed name for categorical list-like columns (PDF: take first element)\n",
    "cat_list_cols = [\"writers\", \"directors\", \"stars\", \"production_companies\"]\n",
    "for col in cat_list_cols:\n",
    "    if col in movies_clean.columns:\n",
    "        movies_clean[col] = (\n",
    "            movies_clean[col]\n",
    "            .str.replace(r\"[\\[\\]']\", \"\", regex=True)\n",
    "            .str.split(\",\").str[0].str.strip()\n",
    "        )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d6d31c92-0642-4418-a8b3-e1df1f372ddb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 10: One-hot encode top-3 for each of those categorical columns (PDF: only top 3)\n",
    "for col in cat_list_cols:\n",
    "    if col in movies_clean.columns:\n",
    "        top3 = movies_clean[col].value_counts().nlargest(3).index\n",
    "        dummies = pd.get_dummies(movies_clean[col], prefix=col)\n",
    "        # keep only columns for top3 (if present)\n",
    "        keep_cols = [f\"{col}_{name}\" for name in top3 if f\"{col}_{name}\" in dummies.columns]\n",
    "        if len(keep_cols) > 0:\n",
    "            dummies = dummies[keep_cols]\n",
    "            movies_clean = pd.concat([movies_clean, dummies], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "d129ba9d-b0af-43b5-b9e0-049b28d3a940",
   "metadata": {},
   "outputs": [],
   "source": [
    "if \"genres\" in movies_clean.columns:\n",
    "    movies_clean[\"main_genre\"] = (\n",
    "        movies_clean[\"genres\"]\n",
    "        .str.replace(r\"[\\[\\]']\", \"\", regex=True)\n",
    "        .str.split(\",\").str[0].str.strip()\n",
    "    )\n",
    "    top15 = movies_clean[\"main_genre\"].value_counts().nlargest(15).index\n",
    "    for g in top15:\n",
    "        movies_clean[g] = (movies_clean[\"main_genre\"] == g).astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "157e2a08-5fc3-4a32-bd11-1ba610dcac1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 12: Feature engineering: year, profit, profit_log and log transforms as PDF\n",
    "movies_clean[\"year\"] = movies_clean[\"release_date\"].dt.year\n",
    "movies_clean[\"profit\"] = movies_clean[\"gross_worldwide\"] - movies_clean[\"budget\"]\n",
    "movies_clean[\"profit_log\"] = np.sign(movies_clean[\"profit\"]) * np.log1p(np.abs(movies_clean[\"profit\"]))\n",
    "\n",
    "# Log transforms\n",
    "movies_clean[\"budget_log\"] = np.log1p(movies_clean[\"budget\"])\n",
    "movies_clean[\"gross_log\"] = np.log1p(movies_clean[\"gross_worldwide\"])\n",
    "movies_clean[\"votes_log\"] = np.log1p(movies_clean[\"votes\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "15256562-4f92-46b1-8990-4fafee76dc9c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['models_pdf_pipeline\\\\scaler_pdf.joblib']"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cell 13: Standardize selected numeric columns (as PDF example)\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaled_cols = [c for c in [\"budget_log\", \"gross_log\", \"votes_log\", \"duration\"] if c in movies_clean.columns]\n",
    "movies_clean[scaled_cols] = scaler.fit_transform(movies_clean[scaled_cols])\n",
    "# save scaler (for later use in app)\n",
    "joblib.dump(scaler, OUT_DIR / \"scaler_pdf.joblib\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7b7e601b-5713-472a-aaa8-f9fd72909adc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature count: 33\n"
     ]
    }
   ],
   "source": [
    "X = movies_clean.select_dtypes(include=[\"float64\", \"int64\", \"int32\", \"bool\"]).copy()\n",
    "remove_cols = [\"gross_worldwide\", \"gross_log\", \"profit\", \"budget\", \"votes\"]\n",
    "X = X.drop(columns=[c for c in remove_cols if c in X.columns], errors=\"ignore\")\n",
    "y = movies_clean[\"gross_log\"]   # target as in PDF (log target)\n",
    "print(\"Feature count:\", X.shape[1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "ead087fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Decision_Tree\n",
      "Training Random_Forest\n",
      "Training SVR\n"
     ]
    }
   ],
   "source": [
    "# Cell 15: train-test split and train three regressors (Decision Tree, RandomForest, SVR) — same as PDF models\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "models = {\n",
    "    \"Decision_Tree\": DecisionTreeRegressor(max_depth=12, random_state=42),\n",
    "    \"Random_Forest\": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),\n",
    "    \"SVR\": (SVR(kernel=\"rbf\", C=10, epsilon=0.1, gamma=\"scale\"))\n",
    "}\n",
    "\n",
    "results = []\n",
    "for name, mdl in models.items():\n",
    "    print(\"Training\", name)\n",
    "    if name == \"SVR\":\n",
    "        # SVR may require scaling — use pipeline-like scaling with StandardScaler for SVR\n",
    "        from sklearn.pipeline import make_pipeline\n",
    "        pipeline = make_pipeline(StandardScaler(), mdl)\n",
    "        pipeline.fit(X_train, y_train)\n",
    "        preds = pipeline.predict(X_test)\n",
    "        joblib.dump(pipeline, OUT_DIR / f\"{name}.joblib\")\n",
    "    else:\n",
    "        mdl.fit(X_train, y_train)\n",
    "        preds = mdl.predict(X_test)\n",
    "        joblib.dump(mdl, OUT_DIR / f\"{name}.joblib\")\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
    "    mae = mean_absolute_error(y_test, preds)\n",
    "    r2 = r2_score(y_test, preds)\n",
    "    results.append((name, rmse, mae, r2))\n",
    "    \n",
    "results_df = pd.DataFrame(results, columns=[\"Model\", \"RMSE\", \"MAE\", \"R2\"]).sort_values(\"RMSE\")\n",
    "results_df\n",
    "# Save metrics\n",
    "results_df.to_csv(OUT_DIR / \"model_metrics.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "6b3e607e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved models and artifacts to models_pdf_pipeline\n"
     ]
    }
   ],
   "source": [
    "# Cell 16: Save final used features list (for Streamlit app)\n",
    "X.columns.to_series().to_csv(OUT_DIR / \"feature_list.csv\", index=False)\n",
    "print(\"Saved models and artifacts to\", OUT_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6bb89679-66d9-4cd5-a8fe-0e6cfcd306e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "154a2ecc-95a5-4bc1-a6a2-c8f63e01d3d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app_streamlit.py\n",
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import silhouette_score\n",
    "from sklearn.cluster import KMeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "833865b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-05 16:29:55.067 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:55.116 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:55.815 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\chels\\anaconda3\\envs\\108758_Pingkan\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-12-05 16:29:55.817 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:55.821 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:56.959 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:56.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:56.964 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:56.969 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:56.982 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.016 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.263 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.268 Session state does not function when running a script without `streamlit run`\n",
      "2025-12-05 16:29:57.278 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.283 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.287 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.301 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.305 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.311 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.317 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.326 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.331 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.335 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.341 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.366 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.369 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.375 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.393 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.448 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.450 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.457 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.463 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.468 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.477 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.481 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.486 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.495 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.509 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.518 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.531 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.536 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.543 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.548 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.552 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.559 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.586 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.595 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.599 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.601 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.603 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.612 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.617 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.628 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.646 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.651 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.661 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.798 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.803 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.808 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.817 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.821 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.828 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.830 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-05 16:29:57.833 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.set_page_config(layout=\"wide\", page_title=\"Movies Final Project — PDF Pipeline\")\n",
    "\n",
    "MODEL_DIR = Path(\"models_pdf_pipeline\")\n",
    "if not MODEL_DIR.exists():\n",
    "    st.error(\"Folder models_pdf_pipeline not found. Jalankan notebook train dan pastikan folder models_pdf_pipeline ada.\")\n",
    "    st.stop()\n",
    "\n",
    "st.title(\"Movies — PDF Pipeline (preprocessing & models same as PDF)\")\n",
    "\n",
    "# Load artifacts\n",
    "scaler = joblib.load(MODEL_DIR / \"scaler_pdf.joblib\")\n",
    "models = {}\n",
    "# load models created (Decision_Tree, Random_Forest, SVR)\n",
    "for name in [\"Decision_Tree\", \"Random_Forest\", \"SVR\"]:\n",
    "    p = MODEL_DIR / f\"{name}.joblib\"\n",
    "    if p.exists():\n",
    "        models[name] = joblib.load(p)\n",
    "\n",
    "kmeans = None\n",
    "kmeans_p = MODEL_DIR / \"kmeans_pdf.joblib\"\n",
    "if kmeans_p.exists():\n",
    "    kmeans = joblib.load(kmeans_p)\n",
    "kmeans_scaler = MODEL_DIR / \"kmeans_scaler.joblib\" if (MODEL_DIR / \"kmeans_scaler.joblib\").exists() else None\n",
    "\n",
    "st.sidebar.header(\"Pilihan\")\n",
    "model_choice = st.sidebar.selectbox(\"Pilih model anggota:\", list(models.keys()))\n",
    "show_kmeans = st.sidebar.checkbox(\"Tampilkan K-Means clustering (PDF style)\", value=True)\n",
    "\n",
    "st.subheader(\"Input fitur untuk prediksi (asumsi fitur sama seperti pipeline PDF)\")\n",
    "# read feature list\n",
    "feat_path = MODEL_DIR / \"feature_list.csv\"\n",
    "if feat_path.exists():\n",
    "    features = list(pd.read_csv(feat_path, header=None).iloc[:,0])\n",
    "else:\n",
    "    # fallback: minimal inputs\n",
    "    features = [\"budget_log\", \"rating\", \"votes_log\", \"duration\", \"year\", \"profit_log\"]\n",
    "\n",
    "# Provide minimal input fields used in PDF top features\n",
    "budget = st.number_input(\"Budget (USD)\", min_value=0.0, value=20000000.0, step=1000.0)\n",
    "rating = st.number_input(\"Rating (0-10)\", min_value=0.0, max_value=10.0, value=6.5, step=0.1)\n",
    "votes = st.number_input(\"Votes\", min_value=0.0, value=12000.0, step=1.0)\n",
    "duration = st.number_input(\"Duration (minutes)\", min_value=0.0, value=100.0, step=1.0)\n",
    "year = st.number_input(\"Year\", min_value=1888, max_value=2100, value=2015, step=1)\n",
    "\n",
    "# Derived features per PDF\n",
    "budget_log = np.log1p(budget)\n",
    "votes_log = np.log1p(votes)\n",
    "profit = 0.0  # unknown -> user can set, but default 0\n",
    "profit_log = np.sign(profit) * np.log1p(np.abs(profit))\n",
    "\n",
    "# Create input vector matching X columns if present\n",
    "# For simplicity, we'll create a small X_top if present in model expectation:\n",
    "X_input = pd.DataFrame({\n",
    "    \"duration\": [duration],\n",
    "    \"rating\": [rating],\n",
    "    \"budget_log\": [budget_log],\n",
    "    \"votes_log\": [votes_log],\n",
    "    \"year\": [year],\n",
    "    \"profit_log\": [profit_log]\n",
    "})\n",
    "# If model expects more columns, Streamlit app can fill missing with 0\n",
    "# Load selected model and predict (handle SVR pipeline)\n",
    "model_obj = models[model_choice]\n",
    "# Ensure columns align: create dataframe with model's expected columns\n",
    "# If model is a pipeline with scaler, it handles scaling internally.\n",
    "try:\n",
    "    # Try direct predict\n",
    "    pred = model_obj.predict(X_input)[0]\n",
    "except Exception:\n",
    "    # Align columns: get feature_list and reindex\n",
    "    try:\n",
    "        feature_list = list(pd.read_csv(MODEL_DIR / \"feature_list.csv\", header=None).iloc[:,0])\n",
    "        X_row = pd.DataFrame(columns=feature_list)\n",
    "        # fill known fields\n",
    "        for c in X_row.columns:\n",
    "            if c in X_input.columns:\n",
    "                X_row.loc[0, c] = X_input.loc[0, c]\n",
    "            else:\n",
    "                X_row.loc[0, c] = 0\n",
    "        pred = model_obj.predict(X_row)[0]\n",
    "    except Exception as e:\n",
    "        st.error(f\"Model prediction error: {e}\")\n",
    "        pred = None\n",
    "\n",
    "if pred is not None:\n",
    "    st.markdown(\"### Hasil prediksi (gross_log)\")\n",
    "    st.write(f\"Prediksi gross_log: **{pred:.4f}**\")\n",
    "    st.write(f\"Untuk konversi ke gross (dollar): exp(pred)-1 = {np.expm1(pred):,.0f}\")\n",
    "\n",
    "# KMeans visualization\n",
    "if show_kmeans and kmeans is not None:\n",
    "    st.markdown(\"## K-Means (PDF pipeline clusters)\")\n",
    "    # load scaled KMeans inputs from movie_clean saved earlier? We'll compute cluster for user input too.\n",
    "    # For visualization, we will load pca values from saved dataset if available; otherwise compute PCA live using saved scaler.\n",
    "    # Here, compute on the original cleaned dataset if saved as CSV (optional). Try to load movies_clean.csv\n",
    "    try:\n",
    "        movies_clean = pd.read_csv(\"movies_clean_saved_for_app.csv\", index_col=0)\n",
    "        k_features = [\"profit_log\", \"budget_log\", \"votes_log\", \"rating\"]\n",
    "        kX = movies_clean[k_features].dropna()\n",
    "        kscaler = joblib.load(MODEL_DIR / \"kmeans_scaler.joblib\")\n",
    "        kX_scaled = kscaler.transform(kX)\n",
    "        # PCA 2D\n",
    "        from sklearn.decomposition import PCA\n",
    "        pca = PCA(n_components=2, random_state=42)\n",
    "        pvals = pca.fit_transform(kX_scaled)\n",
    "        import matplotlib.pyplot as plt\n",
    "        fig, ax = plt.subplots(figsize=(8,6))\n",
    "        labels = kmeans.predict(kX_scaled)\n",
    "        scatter = ax.scatter(pvals[:,0], pvals[:,1], c=labels, cmap=\"tab10\", alpha=0.6)\n",
    "        ax.set_title(\"KMeans clusters (PCA 2D) — PDF pipeline\")\n",
    "        st.pyplot(fig)\n",
    "    except Exception as e:\n",
    "        st.info(\"Data untuk visualisasi KMeans tidak tersedia (movies_clean_saved_for_app.csv). Jika kamu punya file movies_clean_saved_for_app.csv di folder kerja, app akan menampilkan visualization.\")\n",
    "        st.write(\"Error detail (dev):\", e)\n",
    "\n",
    "st.markdown(\"---\")\n",
    "st.caption(\"App ini memuat models & pipeline sesuai contoh PDF (cleaning & feature engineering).\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "82c79a84",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_clean.to_csv(\"movies_clean_saved_for_app.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb3127b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d74edb47-5020-456c-82ea-dfcf1067f607",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (3971350110.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[48], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run app_streamlit.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "179b200b-d25b-4848-9865-45cd98518636",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d48cb26-1d49-4065-a6d5-455fd4b8f0ea",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "d2127cb0",
   "metadata": {},
   "source": [
    "----"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69b7f628",
   "metadata": {},
   "source": [
    "## <div align=\"center\"> Conclusion </div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7ecf70d",
   "metadata": {},
   "source": [
    "### Enter Your Conclusion Here:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df392fe6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "b765bf3f",
   "metadata": {},
   "source": [
    "----"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "650b115b",
   "metadata": {},
   "outputs": [],
   "source": [
    "studentName2 = \"Nama2\"\n",
    "studentNIM2 = \"NIM2\"\n",
    "studentClass2 = \"Class2\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40f84526",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Name: \\t\\t{}\".format(studentName2))\n",
    "print(\"NIM: \\t\\t{}\".format(studentNIM2))\n",
    "print(\"NIM: \\t\\t{}\".format(studentClass2))\n",
    "print(\"End: \\t\\t{}\".format(myDate))\n",
    "print(\"Device ID: \\t{}\".format(myDevice))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14310678",
   "metadata": {},
   "source": [
    "----"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88210d09",
   "metadata": {},
   "source": [
    "## <div align=\"center\"> Reference </div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20664996",
   "metadata": {},
   "source": [
    "### Input Your Reference Here  (Jika ada):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ba2ed2b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "29050624",
   "metadata": {},
   "source": [
    "----"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:108758_Pingkan]",
   "language": "python",
   "name": "conda-env-108758_Pingkan-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
