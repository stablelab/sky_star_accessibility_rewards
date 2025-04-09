from enum import Enum
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Table as SQLAlchemyTable, Column, MetaData, Text, Integer, DateTime, TIMESTAMP, PrimaryKeyConstraint, VARCHAR, Float, Date,Boolean,String,JSON,Numeric
from sqlalchemy import Table as SQLAlchemyTable, Column, MetaData, TEXT, INTEGER, DATETIME, TIMESTAMP, PrimaryKeyConstraint, VARCHAR, FLOAT, DATE, BOOLEAN, JSON, NUMERIC
from sqlalchemy.dialects.postgresql import insert 
from sqlalchemy.sql import text
import concurrent.futures
import re


class DatabaseConnector:
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.engine = self._create_engine()  # Create the engine on initialization

    def _create_engine(self):
        """Creates an SQLAlchemy engine and returns it."""
        try:
            # Create the database URL
            database_url = (
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            )
            # Create and return the SQLAlchemy engine
            return create_engine(database_url)
        except SQLAlchemyError as e:
            print(f"Error creating the engine: {e}")
            return None

    def test_connection(self):
        """Tests the connection to the database."""
        if self.engine:
            try:
                # Attempt a connection to ensure it's working
                with self.engine.connect() as connection:
                    print("Connection successful")
            except SQLAlchemyError as e:
                print(f"Connection test failed: {e}")
        else:
            print("Engine is not initialized")


    def close(self):
        """Closes the connection and disposes of the engine."""
        if self.engine:
            self.engine.dispose()
            print("Connection closed")

class Strategy(Enum):
    IGNORE = lambda t, col: f"{col} = {t}.{col}"
    ADD = lambda t, col: f"{col} = {t}.{col} + EXCLUDED.{col}"
    OVERWRITE = lambda _, col: f"{col} = EXCLUDED.{col}"
    # CONCATENATE = lambda t, col: f"{col} = {t}.{col} || EXCLUDED.{col}"  # Only works for text columns and concatenates them
    

class DtypeConvt:
    def map_dtype_to_sqlalchemy(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype):
            return 'FLOAT'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'BOOLEAN'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'DATETIME'
        elif pd.api.types.is_timedelta64_dtype(dtype):
            return 'DATETIME'
        elif pd.api.types.is_object_dtype(dtype):
            return 'TEXT'
        elif pd.api.types.is_string_dtype(dtype):
            return 'TEXT'
        elif pd.api.types.is_categorical_dtype(dtype):
            return 'TEXT'
        elif pd.api.types.is_float_dtype(dtype):
            return 'FLOAT'
        else:
            return 'TEXT'
    
    def convert_datetime_columns(df):
        for column in df.columns:
            if df[column].dtype == 'object':  
                convert_col = pd.to_datetime(df[column], errors='coerce')
                if convert_col.notna().all():  
                    df[column] = convert_col
        return df
    

    def extract_columns_from_df(df: pd.DataFrame,convert: bool =True) -> dict[str, str]:
        if convert:
            df = DtypeConvt.convert_datetime_columns(df)  # Call the new function to preprocess the DataFrame
        return {col: DtypeConvt.map_dtype_to_sqlalchemy(dtype) for col, dtype in df.dtypes.items()}

class CustomTable:
    def __init__(
        self,
        table_name: str,
        df: pd.DataFrame=None,
        columns: dict[str, str] = None,
        schema_name: str = 'public',
        primary_keys: list[str] = [],
        strategies: dict[str, Strategy] = {},
        default_strategy: Strategy = Strategy.IGNORE,
        engine = None,
        convert: bool = True,
        extract_columns: bool = True,
        snake_case: bool = True):
        
        self.type_mapping = {
            'TEXT': TEXT,
            'VARCHAR': TEXT,
            'CHAR': TEXT,
            'INTEGER': NUMERIC,
            'FLOAT': FLOAT,
            'REAL': FLOAT,
            'DOUBLE': FLOAT,
            'DATE': DATE,
            'TIMESTAMP': TIMESTAMP,
            'DATETIME': TIMESTAMP,
            'BOOLEAN': BOOLEAN,
            'NUMERIC': NUMERIC,
            'JSON': JSON
        }
        self.convert = convert
        self.snake_case = snake_case
        self.table_name: str = table_name
        # Extract columns from DataFrame if provided and extract_columns is True
        if extract_columns and df is not None:
            self.columns : dict[str, str] = DtypeConvt.extract_columns_from_df(df, self.convert)
        
        # if df is not None:
            # self.columns : dict[str, str] = DtypeConvt.extract_columns_from_df(df, self.convert)

        #if columns is are provided
        elif columns is not None:
            self.columns : dict[str, str] = {self.to_snake_case(col): dtype for col, dtype in columns.items()}
        #if neither df nor columns are provided
        else:
            raise ValueError("Either a DataFrame or columns must be provided.")
        # self.columns: dict[str, str] = DtypeConvt.extract_columns_from_df(df,self.convert)  # Extract columns from DataFrame
        self.columns = {self.to_snake_case(col): dtype for col, dtype in self.columns.items()} # Convert column names to snake_case
        self.primary_keys: list[str] = primary_keys
        self.primary_keys = [self.to_snake_case(pk) for pk in self.primary_keys]  # Convert primary keys to snake_case
        self.strategies: dict[str, Strategy] = strategies
        self.strategies = {self.to_snake_case(col): strategy for col, strategy in self.strategies.items()}  # Convert strategies to snake_case
        self.default_strategy: Strategy = default_strategy
        self.engine = engine
        self.schema_name = schema_name
        print(self.columns)

        
        self.__validate()
        
    def create_table(self):
        metadata = MetaData()

        # Type mapping from string to SQLAlchemy types

        # Define columns dynamically with their types and primary key constraints
        columns = []
        for name, dtype in self.columns.items():
            col_type = self.type_mapping.get(dtype.upper())
            if col_type is None:
                raise ValueError(f"Unsupported column type: {dtype}")
            is_primary_key = name in self.primary_keys
            column = Column(name, col_type, nullable=not is_primary_key)
            columns.append(column)

        # Add primary key constraint if applicable
        if self.primary_keys:
            columns.append(PrimaryKeyConstraint(*self.primary_keys))

        # Define and create the table
        my_table = SQLAlchemyTable(self.table_name, metadata, *columns, schema=self.schema_name)

        try:
            metadata.create_all(self.engine)
            print(f"Table '{self.table_name}' created successfully.")
        except SQLAlchemyError as e:
            print(f"Error creating table: {e}")

    def __validate(self):
        column_names = self.columns.keys()
        if len(self.primary_keys) == 0:
            raise ValueError("At least one primary key required")
        if not all(pk in column_names for pk in self.primary_keys):
            raise ValueError("All primary keys must be in columns")
        
        if not all(col in column_names for col in self.strategies.keys()):
            raise ValueError("All strategies must be in columns")
        
        if any(key in self.strategies for key in self.primary_keys):
            raise ValueError("We cannot have primary keys in the strategies list.")
        
    def to_snake_case(self, name: str):
        if self.snake_case:
            # Check if the entire string is uppercase
            if name.isupper():
                return name.lower()
            
            # Convert to snake_case
            s1 = re.sub('([A-Z])', r'_\1', name).lower()
            
            # Handle leading underscore if the string starts with a capital letter
            if name[0].isupper():
                s1 = s1.lstrip('_')
            
            return s1.replace('__', '_')
        return name # Return the name as is if snake_case is False
    
    def add_index(self, columns: list[str],index_name: str=None):
        """Adds an index to the table on the specified columns."""
        if not self.engine:
            raise Exception("Database connection not initialized.")

        # Convert column names to snake_case
        if self.snake_case:
            columns_snake_case = [self.to_snake_case(col) for col in columns]
        else:
            columns_snake_case = columns
        # Validate columns exist in the table
        existing_columns = self.columns.keys()
        if not all(col in existing_columns for col in columns_snake_case):
            raise ValueError(f"Columns {columns_snake_case} are not in the table.")
        # Create the index name if not provided
        if index_name is None:
            index_name = "_".join(columns_snake_case)
        if len(index_name) > 64: # Index name length limit
            index_name = index_name[:64] # takes only the first 64 characters

        try:
            query=f"CREATE INDEX {index_name} ON {self.schema_name}.{self.table_name} ({', '.join(columns_snake_case)});"
            query=text(query)
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    connection.execute(query)
            print(f"Index '{index_name}' created successfully on columns: {', '.join(columns_snake_case)}.")
        except SQLAlchemyError as e:
            print(f"Error creating index: {e}")



    def insert_query(self, rows: list[tuple]):
        column_list = ', '.join([f'"{c}"' for c in self.columns.keys()])
        
        def format_value(value, column_type):
            """Format the value based on its column type."""
            if column_type in ['TEXT', 'TIMESTAMP', 'DATETIME','BOOLEAN']:
                return f"'{value}'"
            return str(value)
        
        # Create the query
        query = f"INSERT INTO {self.schema_name}.{self.table_name} ({column_list}) VALUES "
        
        # Format each row based on its column type
        row_values_list = [
            f"({', '.join(format_value(value, list(self.columns.values())[i]) for i, value in enumerate(row))})"
            for row in rows
        ]
        
        row_values = ', '.join(row_values_list)
        
        query += row_values
        
        # Clean up query
        query = query.rstrip(', ')

        if len(self.primary_keys) > 0 and (len(self.strategies.keys()) > 0 or self.default_strategy != Strategy.IGNORE):
            # Build the conflict clause
            conflict_clause = " ON CONFLICT (" + ', '.join(f'"{key}"' for key in self.primary_keys) + ") DO UPDATE SET "

            columns_strategies = [
                                    self.strategies[column](self.table_name, f'"{column}"') if column in self.strategies else
                                    (self.default_strategy(self.table_name, f'"{column}"') if 
                                        (len(self.strategies.keys()) == 0 and 
                                        (self.columns[column] == 'INTEGER' or self.columns[column] == 'FLOAT') and 
                                        self.default_strategy == Strategy.ADD) 
                                    else 
                                    (self.default_strategy(self.table_name, f'"{column}"') if 
                                        (len(self.strategies.keys()) == 0 and self.default_strategy == Strategy.OVERWRITE) 
                                    else None))
                                    for column in self.columns.keys() if column not in self.primary_keys
                                ]


            # Filter out any None values (in case a column didn't match any strategy)
            columns_strategies = [strategy for strategy in columns_strategies if strategy is not None]

            # Now include these strategies in the conflict clause
            query += conflict_clause + ', '.join(columns_strategies)

            # print("Conflict clause", conflict_clause, columns_strategies)

            # conflict_clause = f" ON CONFLICT ({', '.join(self.primary_keys)}) DO UPDATE SET "
            # columns_strategies = [strategy(self.table_name, column) for column, strategy in self.strategies.items()]
            # query += conflict_clause + ', '.join(columns_strategies)
            # print("Conflict clause",conflict_clause,columns_strategies)
        
        
        query += ';'
        # print(query)
        return text(query)
    
    def chunker(self, data, batch_size):

        if isinstance(data, pd.DataFrame):
            # Ensure DataFrame contains all required columns
            # data.columns = [self.to_snake_case(col) for col in data.columns]
            missing_cols = set(self.columns.keys()) - set(data.columns)

            if missing_cols:
                raise ValueError(f"DataFrame is missing columns: {', '.join(missing_cols)}")

            # set nans to null

            data = data[self.columns.keys()]# Reorder columns to match the table schema
            data.fillna('null', inplace=True)
            
            
            data = data.values.tolist()# Convert DataFrame to list of tuples
                
        elif isinstance(data, list):
            if len(data) > 0 and len(data[0]) != len(self.columns):
                raise ValueError("List of tuples does not match the table schema.")
            return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        else:
            raise ValueError("Data must be either a list of tuples or a pandas DataFrame")
        
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


    def insert(self, data, batch_size: int=1000, num_threads: int=1):
        if self.snake_case is True:
            data.columns = [self.to_snake_case(col) for col in data.columns]
        # print("dataset columns",data.columns)  
        if not self.engine:
            raise Exception("Database connection not initialized.")
        chunks = self.chunker(data, batch_size)

        def insert_chunk(chunk):
            query = self.insert_query(chunk)
            try:
                with self.engine.connect() as connection:
                    with connection.begin() as transaction:
                        connection.execute(query)
            except SQLAlchemyError as e:
                print(f"Error executing query for chunk:")
                print(e)
            return len(chunk)

        total_rows_inserted = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {executor.submit(insert_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    rows_inserted = future.result() 
                    total_rows_inserted += rows_inserted
                except Exception as exc:
                    print(f"Chunk generated an exception:")
                    print(exc)

        print(f"Total rows inserted: {total_rows_inserted}")


    def print_table_schema(self):
        """Prints the schema of the table."""
        if not self.engine:
            print("Database connection not initialized.")
            return
        
        inspector = inspect(self.engine)
        columns = inspector.get_columns(self.table_name)
        
        print(f"Table schema for '{self.table_name}':")
        for column in columns:
            print(f"Column Name: {column['name']}, Type: {column['type']}, Nullable: {column['nullable']}")
        print()  # Add an empty line for better readability

    def print_table_data(self):
        """Prints the data from the table."""
        if not self.engine:
            print("Database connection not initialized.")
            return
        
        query1 = f"SELECT * FROM {self.table_name} LIMIT 1000;"
        
        try:
            with self.engine.connect() as connection:
                print(f"Executing query: {query1}")  # Print the query for debugging
                result = connection.execute(text(query1))
                print(f"Data from table '{self.table_name}':")
                df = pd.read_sql_table(self.table_name, self.engine)
                print(df)
        except SQLAlchemyError as e:
            print(f"Error fetching data: {e}")

    def print_table_info(self):
        """Prints the data from the table."""
        if not self.engine:
            print("Database connection not initialized.")
            return        
        try:
            with self.engine.connect() as connection:
                # Get the total count of rows
                count_query = f"SELECT COUNT(*) FROM {self.schema_name}.{self.table_name};"
                print(f"Executing query: {count_query}")  # Print the query for debugging
                row_count_result = connection.execute(text(count_query)).fetchone()
                row_count = row_count_result[0] if row_count_result else 0
                
                # Get the number of columns
                # Fetch a single row to determine column count
                # sample_query = f"SELECT * FROM {self.table_name} LIMIT 1;"
                # print(f"Executing query: {sample_query}")  # Print the query for debugging
                # sample_result = connection.execute(text(sample_query))
                # columns = sample_result.keys()
                # column_count = len(columns)
                
                print(f"Data from table '{self.table_name}':")
                print(f"Number of rows: {row_count}")
                # print(f"Number of columns: {column_count}")
                
        except SQLAlchemyError as e:
            print(f"Error fetching data: {e}")