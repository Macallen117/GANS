import json
import csv
import ast

def do_something(in_file, out_file):
    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        header = ["address", "timestamp", "value"]
        writer = csv.DictWriter(out_f, fieldnames=header, delimiter=';')
        writer.writeheader()

        for line in in_f:
            # print(line)
            # parsed_line = ast.literal_eval(line)
            # obj = json.loads(parsed_line)

            obj = json.loads(line)
            # obj = json.loads(eval(line))
            # obj = json.dumps(line)
            # Hier wird die ganze Datei durchlaufen,
            # daher auch .load's' und nicht .load .loads macht einen String aus der Datei

            # # mehrere Schleifen um die richtigen Werte zu erhalten
            # for td in obj["TraceData"]:
            #   for dp in td["body"]["datapoint"]:
            #     timestamp = dp["timestamp"],
            #     address = dp["address"]
            #     values = dp["content"][0]["value"]
            #     row = {
            #         "address": address,
            #         "timestamp": timestamp,
            #         "value": values
            #     }
            #     writer.writerow(row) #Ausgabe der Adress, Timestamp und Values in den Zeilen

if __name__ == '__main__':
    do_something("085240.json", "085240.csv")