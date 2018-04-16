from DataFrame import DataFrame


class CsvReader(object):

    @staticmethod
    def create_from_csv(csv_path, sep=";"):
        data_read = CsvReader.__read_from_file(csv_path)
        unprocessed_data = CsvReader.__create_data_frame_from_read_data(data_read, sep=sep)
        
        lab = CsvReader.create_labels(unprocessed_data)
        dat = CsvReader.create_data(unprocessed_data)
        
        df = DataFrame(labels=lab, data=dat)
        return df

    @staticmethod
    def __read_from_file(file_path):
        with open(file_path, 'r') as file:
            readlines = []
            for line in file.readlines():
                readlines.append(line[0:-1])
            return readlines

    @staticmethod
    def __create_data_frame_from_read_data(read_lines, sep=";"):
        
        divided_data = [line.split(sep) for line in read_lines]
        
        #remove the first line (header)
        #divided_data = [line.split(sep) for line in read_lines[1:]]
        
        return divided_data

    @staticmethod
    def create_labels(unprocessed_data):
        return [x.strip() for x in unprocessed_data[0]]
        
        #return [x.strip() for x in unprocessed_data[:, 4]]

    @staticmethod
    def create_data(unprocessed_data):
        return unprocessed_data[1:]
        
        #return unprocessed_data[0:3]


