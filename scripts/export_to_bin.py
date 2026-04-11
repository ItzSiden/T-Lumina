import torch
import struct

INPUT = "tlumina_packed_5in8.pt"
OUTPUT = "tlumina_model.bin"

state = torch.load(INPUT, map_location="cpu")

with open(OUTPUT, "wb") as f:
    for name, obj in state.items():
        print("Writing:", name)

        name_bytes = name.encode()
        f.write(struct.pack("I", len(name_bytes)))
        f.write(name_bytes)

        # ----------------------------
        # TYPE DISPATCH
        # ----------------------------

        # 1 = FP16 Tensor
        # 2 = UINT8 Tensor
        # 3 = Shape (list of ints)
        # 4 = Single integer

        if isinstance(obj, torch.Tensor):

            if obj.dtype == torch.float16:
                f.write(struct.pack("I", 1))
                data = obj.numpy().tobytes()

            elif obj.dtype == torch.uint8:
                f.write(struct.pack("I", 2))
                data = obj.numpy().tobytes()

            else:
                raise ValueError(f"Unsupported tensor dtype: {obj.dtype}")

        elif isinstance(obj, torch.Size):

            f.write(struct.pack("I", 3))
            shape_list = list(obj)
            data = struct.pack("I"*len(shape_list), *shape_list)

        elif isinstance(obj, int):

            f.write(struct.pack("I", 4))
            data = struct.pack("I", obj)

        else:
            raise ValueError(f"Unsupported type: {type(obj)}")

        f.write(struct.pack("I", len(data)))
        f.write(data)

print("Binary export complete.")