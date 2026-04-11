import struct

with open("./tlumina_model.bin", "rb") as f:
    count = 0
    while count < 40:
        len_bytes = f.read(4)
        if not len_bytes or len(len_bytes) < 4: break
        name_len = struct.unpack("I", len_bytes)[0]
        name = f.read(name_len).decode("utf-8")
        type_val = struct.unpack("I", f.read(4))[0]
        
        if type_val in (1, 2):
            data_len = struct.unpack("I", f.read(4))[0]
            f.seek(data_len, 1)
        elif type_val in (4, 5):
            f.read(4)  # len field
            f.read(4)  # value
        
        print(f"[{type_val}] {name}")
        count += 1