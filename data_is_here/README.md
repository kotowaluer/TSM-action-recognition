Step 0.
    ignore 3 files: dataset_hands.py, hand_position_extractor.py and TSN_with_hands.py.
    this is my implement for adding hands info into model.

Step 1.
    Split the video into frames and name them by '{video_name}_{10d}.jpg'.
    For example, a video call example.mp4 with 250 frames, 
    then there should be 250 images called example_0000000000.jpg to example_0000000249.jpg
    One video one folder. Put all example_{10d}.jpg into a folder called 'example'
    
Step 2.
    Generate the category.txt which is a map from string label to number label.
    (just one line one category name, no need for '0 pick')
    For example in the txt:
    pick up
    put down
    grasp
    ...
    
Step 3.
    Annotations generation.
    For one training data, you need mark its '{saving path} {start frame index} {how many frames} {number label}'.
    For example:
    frames/example0/example0 473 7 1
    frames/example1/example1 1264 43 4
    frames/example2/example2 498 145 5
    frames/example3/example3 117 16 7
    frames/example4/example4 287 30 2
    ...
    Then you can split them into two txt called 'train_rgb.txt' and 'validation_rgb.txt'.

Step 4.
    Modify the code in 'ops/data_config.py'.
    Check the code in line 7(:path_to_data) and line 12(:root_data).

Step 5.
    run with script:
    python main.py Assembly101 RGB --arch resnet101 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 250 --batch-size 8 -j 16 --dropout 0.25 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb

Step 6.
    I think you may have some path issue since I did not optimize them currently. So you can see I used {video_name}/{video_name} and the path of frames.

Step 7.
    I did not put too many efforts on param modification so if you many have some try on params.