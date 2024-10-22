import hashlib

print('=========input')
for i in range(8):
    print('>>>>>>>>>>>>>>>>>',i)
    torch_md5 = hashlib.md5(open('torch_bin/rank'+str(i)+'_input.bin','rb').read()).hexdigest()
    paddle_md5 = hashlib.md5(open('paddle_bin/rank'+str(i)+'_input.bin','rb').read()).hexdigest()
    print(torch_md5)
    print(paddle_md5)
    print(torch_md5 == paddle_md5)

print('=========weight')
for i in range(8):
    print('>>>>>>>>>>>>>>>>>',i)
    torch_md5 = hashlib.md5(open('torch_bin/rank'+str(i)+'_weight.bin','rb').read()).hexdigest()
    paddle_md5 = hashlib.md5(open('paddle_bin/rank'+str(i)+'_weight.bin','rb').read()).hexdigest()
    print(torch_md5)
    print(paddle_md5)
    print(torch_md5 == paddle_md5)

print('=========output')
for i in range(8):
    print('>>>>>>>>>>>>>>>>>',i)
    torch_md5 = hashlib.md5(open('torch_bin/rank'+str(i)+'_out.bin','rb').read()).hexdigest()
    paddle_md5 = hashlib.md5(open('paddle_bin/rank'+str(i)+'_out.bin','rb').read()).hexdigest()
    print(torch_md5)
    print(paddle_md5)
    print(torch_md5 == paddle_md5)
