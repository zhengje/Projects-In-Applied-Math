import Information from "./information"

export default function Modal({closeModal, data}) {
    const listItems = Array.isArray(data) ? data : []

    return (
        <div className='modalBackground'>
            <div className='modalContainer'>
                <div className='titleCloseBtn'>
                    <button className='titleCloseBtn' onClick={() => {closeModal(false)}}>❌</button>
                </div>
                <div className='title'>Recommendations for KNN Method</div>
                <div className='body'>

                    <u1>
                    {listItems.map((item, index) => (
                        <Information id='list-div'
                        key={item.title || index}
                        rank={item.rank}
                        price={item.price}
                        ratings={item.ratings}
                        // genres={}
                        />
                    ))}
                    </u1>
                </div>
            </div>
        </div>
    )
}