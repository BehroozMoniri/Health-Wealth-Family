// Create the "Back to Top" button dynamically
const topBtn = document.createElement("button");
topBtn.innerText = "↑ Top";
topBtn.id = "backToTop";
document.body.appendChild(topBtn);

// Style the button using JS so you don't have to add more CSS
Object.assign(topBtn.style, {
    position: "fixed",
    bottom: "30px",
    right: "30px",
    padding: "10px 15px",
    backgroundColor: "#2b6cb0",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    display: "none",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
    fontSize: "16px",
    zIndex: "1000"
});

// Show button when user scrolls down 300px
window.onscroll = function() {
    if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
        topBtn.style.display = "block";
    } else {
        topBtn.style.display = "none";
    }
};

// Smooth scroll to top when clicked
topBtn.addEventListener("click", function() {
    window.scrollTo({ top: 0, behavior: "smooth" });
});