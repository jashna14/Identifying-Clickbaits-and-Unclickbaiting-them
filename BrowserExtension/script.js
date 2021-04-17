
document.addEventListener('DOMContentLoaded', function() {
    var tempButton=document.getElementById("trigger");
    tempButton.addEventListener("click",executeDetection);
});

function executeDetection(){
    var temp=document.getElementById("trigger");
    console.log("innn")
    if(temp.innerText!=='Loading...'){
        temp.innerHTML="<span class=\"spinner-border spinner-border-sm\"></span> \n  Loading...";
        chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
            let url = tabs[0].url;
                
                let link='http://localhost:5000/titleandcontent';
                const params = {"url":url};
                console.log(params)

                fetch(link, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(params)
                })
                .then(r=>r.text())
                .then(res=>{
                    data=JSON.parse(res)
                    console.log(data)
                    const params1 = {"title": JSON.parse(res)["title"] ,
                                    "content": JSON.parse(res)["content"]
                        };
                    let link1='http://localhost:5000/identify';
                    fetch(link1, {
                        method: 'POST',
                        headers: {
                        'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(params1)
                    })
                    .then(r=>r.text())
                    .then(res=>{
                        data=JSON.parse(res)
                        console.log(data)
                        clickbait = JSON.parse(res)["clickbait"]
                        newtitle = JSON.parse(res)["newtitle"]

                        console.log(clickbait)
                        console.log(newtitle)

                        if(clickbait=="yes"){
                            var per = document.getElementById("clickbait")
                            per.innerHTML="Yes,Title is clickbaity"
                            var tit = document.getElementById("newtitle")
                            tit.innerHTML="Newtitle :-"+ newtitle
                        }
                        else{
                            var per = document.getElementById("clickbait")
                            per.innerHTML="No,Title is unclickbaity"
                        }

                    })

                //     var table = document.getElementById("results")
                //     var row = table.insertRow(0)
                //     row.insertCell(0).innerHTML="Tweet text";
                //     row.insertCell(1).innerHTML="Controversial";
                    
                //     for(const [tweetText,contro] of Object.entries(data)){
                //         var row = table.insertRow(-1)

                //         if(contro == true){
                //         	temporary = "<span style ='background-color:red;'>"+contro+"</style>"
                //         }else{
                //         	temporary = "<span style ='background-color:green;'>"+contro+"</style>"
                //         }
                //         row.insertCell(0).innerHTML=tweetText;
                //         row.insertCell(1).innerHTML=temporary;
                    
                //         }
                    })
            // }
        });
    }
    temp.innerHTML="  <span></span>\n  GenerateNewTitle!";
    
}
