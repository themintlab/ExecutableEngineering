import struct

def python_internal_binary(num):
    # Pack the float into 8 bytes using IEEE 754 double-precision ('d')
    # The '!' ensures network byte order (big-endian) so bits read left-to-right
    packed_bytes = struct.pack('!d', float(num))
    
    # Unpack those bytes as a single 64-bit unsigned integer ('Q')
    packed_int, = struct.unpack('!Q', packed_bytes)
    
    # Format the integer as a 64-bit binary string, padded with leading zeros
    binary_str = f"{packed_int:064b}"
    
    # Split into the IEEE 754 components for readability
    sign = binary_str[0]         # 1 bit
    exponent = binary_str[1:12]  # 11 bits
    fraction = binary_str[12:]   # 52 bits
    
    return f"{sign} {exponent} {fraction}"

def decimal_to_binary(num, precision=20):
    sign = "-" if num < 0 else ""
    num = abs(num)
    int_part = int(num)
    frac = num - int_part
    int_str = bin(int_part)[2:]
    
    frac_part = ""
    for _ in range(precision):
        if frac == 0: 
            break
        frac *= 2
        bit = int(frac)
        frac_part += str(bit)
        frac -= bit
        
    if not frac_part:
        frac_part = "0"
        
    return f"{sign}{int_str}.{frac_part}"
