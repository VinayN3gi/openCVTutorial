import math
from Crypto.Cipher import DES, AES
from Crypto.Util.Padding import pad, unpad
import binascii


# ----------------------- Substitution Ciphers -----------------------

def caesar_cipher(text, key, mode='encrypt'):
    result = ""
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shift = key if mode == 'encrypt' else -key
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result


def monoalphabetic_cipher(text, key, mode='encrypt'):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if len(key) != 26:
        raise ValueError("Key must be 26 characters.")
    enc_map = str.maketrans(alphabet + alphabet.upper(), key.lower() + key.upper())
    dec_map = str.maketrans(key.lower() + key.upper(), alphabet + alphabet.upper())
    return text.translate(enc_map if mode == 'encrypt' else dec_map)


def vigenere_cipher(text, key, mode='encrypt'):
    res, k = "", key.lower()
    j = 0
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            shift = ord(k[j % len(k)]) - ord('a')
            shift = shift if mode == 'encrypt' else -shift
            res += chr((ord(c) - base + shift) % 26 + base)
            j += 1
        else:
            res += c
    return res


# ----------------------- Playfair Cipher -----------------------

def generate_playfair_matrix(key):
    key = key.lower().replace(" ", "").replace("j", "i")
    chars = []
    for c in key:
        if c not in chars:
            chars.append(c)
    for c in "abcdefghiklmnopqrstuvwxyz":
        if c not in chars:
            chars.append(c)
    return [chars[i:i + 5] for i in range(0, 25, 5)]


def find_char(matrix, ch):
    for r, row in enumerate(matrix):
        for c, v in enumerate(row):
            if v == ch:
                return r, c
    return -1, -1


def playfair_cipher(text, key, mode='encrypt'):
    matrix = generate_playfair_matrix(key)
    text = text.lower().replace(" ", "").replace("j", "i")
    if mode == 'encrypt':
        i = 0
        while i < len(text) - 1:
            if text[i] == text[i + 1]:
                text = text[:i + 1] + 'x' + text[i + 1:]
            i += 2
        if len(text) % 2:
            text += 'x'
    pairs = [text[i:i + 2] for i in range(0, len(text), 2)]
    res = ""
    for p in pairs:
        r1, c1 = find_char(matrix, p[0])
        r2, c2 = find_char(matrix, p[1])
        shift = 1 if mode == 'encrypt' else -1
        if r1 == r2:
            res += matrix[r1][(c1 + shift) % 5] + matrix[r2][(c2 + shift) % 5]
        elif c1 == c2:
            res += matrix[(r1 + shift) % 5][c1] + matrix[(r2 + shift) % 5][c2]
        else:
            res += matrix[r1][c2] + matrix[r2][c1]
    return res, matrix


# ----------------------- Transposition Ciphers -----------------------

def rail_fence_cipher(text, key, mode='encrypt'):
    if mode == 'encrypt':
        rails, row, step = [''] * key, 0, 1
        for c in text:
            rails[row] += c
            row += step
            if row == 0 or row == key - 1:
                step *= -1
        return ''.join(rails)
    rail_lens, row, step = [0] * key, 0, 1
    for _ in text:
        rail_lens[row] += 1
        row += step
        if row == 0 or row == key - 1:
            step *= -1
    rails, start = [], 0
    for ln in rail_lens:
        rails.append(list(text[start:start + ln]))
        start += ln
    res, row, step = "", 0, 1
    for _ in text:
        res += rails[row].pop(0)
        row += step
        if row == 0 or row == key - 1:
            step *= -1
    return res


def columnar_transposition_cipher(text, key, mode='encrypt'):
    num_cols, key_map = len(key), sorted([(k, i) for i, k in enumerate(key)])
    if mode == 'encrypt':
        rows = math.ceil(len(text) / num_cols)
        text = text.ljust(rows * num_cols, 'x')
        m = [list(text[i:i + num_cols]) for i in range(0, len(text), num_cols)]
        return ''.join(m[r][i] for _, i in key_map for r in range(rows))
    rows = math.ceil(len(text) / num_cols)
    short_cols = num_cols * rows - len(text)
    m = [[''] * num_cols for _ in range(rows)]
    text_i = 0
    for _, i in key_map:
        ln = rows - (1 if i >= num_cols - short_cols else 0)
        for r in range(ln):
            m[r][i] = text[text_i]
            text_i += 1
    return ''.join(''.join(r) for r in m)


# ----------------------- DES Demo -----------------------

IP_TABLE = [58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7]

E_TABLE = [32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13,
           12, 13, 14, 15, 16, 17, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23,
           24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1]


def permute(block, table):
    return "".join(block[i - 1] for i in table)


def des_one_round_demo(plaintext_hex, key_hex):
    print("--- DES One Round ---")
    pt = bin(int(plaintext_hex, 16))[2:].zfill(64)
    key = bin(int(key_hex, 16))[2:].zfill(64)
    round_key = key[:48]
    ptext = permute(pt, IP_TABLE)
    L0, R0 = ptext[:32], ptext[32:]
    exp_R0 = permute(R0, E_TABLE)
    xored = bin(int(exp_R0, 2) ^ int(round_key, 2))[2:].zfill(48)
    sbox_out = "11010010101001011111000010011011"
    R1 = bin(int(L0, 2) ^ int(sbox_out, 2))[2:].zfill(32)
    print(f"L1={R0}, R1={R1}")

    cipher = DES.new(binascii.unhexlify(key_hex), DES.MODE_ECB)
    enc = cipher.encrypt(binascii.unhexlify(plaintext_hex))
    dec = cipher.decrypt(enc)
    print(f"Encrypted: {binascii.hexlify(enc).decode()}")
    print(f"Decrypted: {binascii.hexlify(dec).decode()}")


# ----------------------- AES Demo -----------------------

def bytes_to_matrix(b): return [list(b[i:i + 4]) for i in range(0, 16, 4)]


def print_state(name, s):
    print(f"\n{name}:")
    for r in s:
        print(" ".join(f"{v:02x}" for v in r))


def aes_encryption_demo(pt_bytes, key_bytes):
    s = bytes_to_matrix(pt_bytes)
    k = bytes_to_matrix(key_bytes)
    for r in range(4):
        for c in range(4):
            s[r][c] ^= k[r][c]
    print_state("After AddRoundKey", s)
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    enc = cipher.encrypt(pad(pt_bytes, AES.block_size))
    dec = unpad(cipher.decrypt(enc), AES.block_size)
    print(f"\nAES Encrypted: {enc.hex()}")
    print(f"AES Decrypted: {dec.decode()}")


# ----------------------- Run All -----------------------

def main():
    print("=" * 50)
    print("SUBSTITUTION CIPHERS")
    text = "Meet me at midnight"
    print("\nCaesar:", caesar_cipher(text, 5))
    print("Decrypted:", caesar_cipher(caesar_cipher(text, 5), 5, 'decrypt'))

    key_mono = "QWERTYUIOPASDFGHJKLZXCVBNM"
    print("\nMonoalphabetic:", monoalphabetic_cipher(text, key_mono))
    print("Decrypted:", monoalphabetic_cipher(monoalphabetic_cipher(text, key_mono), key_mono, 'decrypt'))

    print("\nVigenere:", vigenere_cipher(text, "SECRET"))
    print("Decrypted:", vigenere_cipher(vigenere_cipher(text, "SECRET"), "SECRET", 'decrypt'))

    print("\n" + "=" * 50)
    print("PLAYFAIR CIPHER")
    enc, mat = playfair_cipher("defend the castle", "fortify")
    print(f"Encrypted: {enc}")

    print("\n" + "=" * 50)
    print("TRANSPOSITION CIPHERS")
    print("Rail Fence:", rail_fence_cipher("weattackatdawn", 3))
    print("Columnar:", columnar_transposition_cipher("defendtheeastwall", "KEY"))

    print("\n" + "=" * 50)
    print("DES DEMO")
    des_one_round_demo("0123456789ABCDEF", "133457799BBCDFF1")

    print("\n" + "=" * 50)
    print("AES DEMO")
    aes_encryption_demo(b"Data encryption!", b"StrongAESKey1234")


if __name__ == "__main__":
    main()
