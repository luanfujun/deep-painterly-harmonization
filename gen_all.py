import os

numImgs = 35
gpu=0

if os.path.exists('results') == 0:
	os.mkdir('results')

for idx in range(3, numImgs):
	print 'Working on index = ', idx
	part_cmd1 =' th neural_gram.lua '\
			   ' -content_image data/' + str(idx) + '_naive.jpg  '\
			   ' -style_image   data/' + str(idx) + '_target.jpg '\
			   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
			   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
			   ' -gpu ' + str(gpu) + ' -original_colors 0 -image_size 700 '\
			   ' -output_image  results/' + str(idx) + '_inter_res.jpg'\
			   ' -print_iter 100 -save_iter 100 && '

	part_cmd2 =' th neural_paint.lua '\
			   ' -content_image data/' + str(idx) + '_naive.jpg '\
			   ' -style_image   data/' + str(idx) + '_target.jpg '\
			   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
			   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
			   ' -cnnmrf_image  results/' + str(idx) + '_inter_res.jpg  '\
			   ' -gpu ' + str(gpu) + ' -original_colors 0 -image_size 700 '\
			   ' -index ' + str(idx) + ' -wikiart_fn data/wikiart_output.txt '\
			   ' -output_image  results/' + str(idx) + '_final_res.jpg' \
			   ' -print_iter 100 -save_iter 100 '\
			   ' -num_iterations 1000 ' 
	cmd = part_cmd1 + part_cmd2
	os.system(cmd)
