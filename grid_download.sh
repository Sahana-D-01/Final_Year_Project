mkdir "gridcorpus"
cd "gridcorpus"
mkdir "raw" "audio" "video"
cd "raw" && mkdir "audio" "video"

for ((i=1; i<=1; i++))
do
    printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
    
    #download the audio of the ith speaker
    cd "audio" && curl "https://spandh.dcs.shef.ac.uk/gridcorpus/s$i/audio/s$i.tar" > "s$i.tar" && cd ..
    cd "video" && curl "https://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s1.mpg_vcd.zip" > "s1.zip" && cd ..

    #if [[ $3 == "y" ]]
    #then
        unzip -q "video/s$i.zip" -d "../video"
        tar -xf "audio/s$i.tar" -C "../audio"
    #fi
done
