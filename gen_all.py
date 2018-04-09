import os

numImgs = 34
gpu=0

if os.path.exists('results') == 0:
	os.mkdir('results')

for idx in range(0, numImgs):
	# part_cmd1 =' th neural_gram.lua '\
	# 		   ' -content_image data/' + str(idx) + '_naive.jpg  '\
	# 		   ' -style_image   data/' + str(idx) + '_target.jpg '\
	# 		   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
	# 		   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
	# 		   ' -gpu ' + str(gpu) + ' -original_colors 0 -image_size 700 '\
	# 		   ' -output_image  ' + str(idx) + '_intermediate.jpg -print_iter 100 -save_iter 100 '

	# Note: For speed we omit intermediate harmonization.
	part_cmd2 =' th neural_paint.lua '\
			   ' -content_image data/' + str(idx) + '_naive.jpg '\
			   ' -style_image   data/' + str(idx) + '_target.jpg '\
			   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
			   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
			   ' -gpu ' + str(gpu) + ' -original_colors 0 -image_size 700 '\
			   ' -index ' + str(idx) + ' -wikiart_fn data/wikiart_output.txt '\
			   ' -output_image  results/' + str(idx) + '_res.jpg' \
			   ' -print_iter 100 -save_iter 100 '\
			   ' -num_iterations 1000 ' 

	os.system(part_cmd2)
