#Use this script to convert all images that do not have a .GUI pair

List = open('./451-500/rename.txt')
List2 = (s.strip() for s in List)
# Loop through the list and create a file with
for item in List2:
    open('./451-500/%s'%(item,), 'w')