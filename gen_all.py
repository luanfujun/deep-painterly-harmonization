import os
import math

numImgs = 35
# numGpus = 16 
numGpus = 1

if os.path.exists('results') == 0:
	os.mkdir('results')

N = int(math.ceil(float(numImgs)/numGpus))

for j in range(1, numGpus+1):
	cmd = ''
	for i in range(1, N+1):
		idx = (i-1) * numGpus + (j-1)
		if idx >= 0 and idx < numImgs:
			print('Working on image idx = ', idx)
			part_cmd1 =' th neural_gram.lua '\
					   ' -content_image data/' + str(idx) + '_naive.jpg  '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
					   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
					   ' -gpu ' + str(j-1) + ' -original_colors 0 -image_size 700 '\
					   ' -output_image  results/' + str(idx) + '_inter_res.jpg'\
					   ' -print_iter 100 -save_iter 100 && '
			part_cmd2 =' th neural_paint.lua '\
					   ' -content_image data/' + str(idx) + '_naive.jpg '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
					   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
					   ' -cnnmrf_image  results/' + str(idx) + '_inter_res.jpg  '\
					   ' -gpu ' + str(j-1) + ' -original_colors 0 -image_size 700 '\
					   ' -index ' + str(idx) + ' -wikiart_fn data/wikiart_output.txt '\
					   ' -output_image  results/' + str(idx) + '_final_res.jpg' \
					   ' -print_iter 100 -save_iter 100 '\
					   ' -num_iterations 1000 &&' 
			cmd = cmd + part_cmd1 + part_cmd2
	cmd = cmd[1:len(cmd)-1]
	print(cmd)
	os.system(cmd)
